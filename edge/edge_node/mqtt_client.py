"""
MQTT 客户端模块
负责边缘节点与云端/中控的数据通信
支持断网缓存、QoS 1 可靠传输
"""
import json
import logging
import time
import threading
from typing import Optional, Dict, List, Callable
from queue import Queue, Full

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed, MQTT disabled")


class EdgeMqttClient:
    """
    边缘 MQTT 客户端
    
    功能:
    - 连接 MQTT Broker
    - 发布结构化识别结果
    - 断网时本地队列缓存
    - 自动重连与恢复
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "zhishitong/node",
        client_id: str = "edge-01",
        qos: int = 1,
        reconnect_interval: int = 5,
        max_queue_size: int = 5000
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix
        self.client_id = client_id
        self.qos = qos
        self.reconnect_interval = reconnect_interval
        
        self.client = None
        self.connected = False
        self._lock = threading.Lock()
        
        # 离线消息队列
        self.offline_queue = Queue(maxsize=max_queue_size)
        self._flush_thread = None
        self._running = False
        
        if MQTT_AVAILABLE:
            self._init_client()
        else:
            logger.warning("MQTT client initialized in NO-OP mode")
    
    def _init_client(self):
        """初始化 MQTT 客户端"""
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        # 设置遗嘱消息 (LWT)
        will_topic = f"{self.topic_prefix}/status"
        will_payload = json.dumps({
            "client_id": self.client_id,
            "status": "offline",
            "timestamp": time.time()
        })
        self.client.will_set(will_topic, will_payload, qos=1, retain=True)
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"MQTT connected to {self.broker_host}:{self.broker_port}")
            
            # 发布在线状态
            self._publish_status("online")
            
            # 启动 flush 线程
            self._start_flush_thread()
        else:
            logger.error(f"MQTT connection failed, rc={rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开回调"""
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT unexpected disconnect, rc={rc}")
    
    def _on_publish(self, client, userdata, mid):
        """发布回调"""
        logger.debug(f"Message {mid} published")
    
    def connect(self) -> bool:
        """连接到 Broker"""
        if not MQTT_AVAILABLE or self.client is None:
            return False
        
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            
            # 等待连接建立
            for _ in range(10):
                if self.connected:
                    return True
                time.sleep(0.5)
            
            return self.connected
        except Exception as e:
            logger.error(f"MQTT connect error: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        
        if self._flush_thread:
            self._flush_thread.join(timeout=2)
        
        if self.client:
            self._publish_status("offline")
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("MQTT disconnected")
    
    def publish_detection(self, result: Dict) -> bool:
        """
        发布识别结果
        
        Args:
            result: 流水线输出的结构化结果
            
        Returns:
            是否成功发送或入队
        """
        payload = json.dumps(result, ensure_ascii=False, default=str)
        topic = f"{self.topic_prefix}/detection"
        
        return self._publish_or_queue(topic, payload)
    
    def publish_heartbeat(self, stats: Dict):
        """发布心跳/统计信息"""
        payload = json.dumps({
            "client_id": self.client_id,
            "timestamp": time.time(),
            "stats": stats,
            "queue_size": self.offline_queue.qsize()
        })
        topic = f"{self.topic_prefix}/heartbeat"
        self._publish_or_queue(topic, payload)
    
    def _publish_status(self, status: str):
        """发布节点状态"""
        payload = json.dumps({
            "client_id": self.client_id,
            "status": status,
            "timestamp": time.time()
        })
        topic = f"{self.topic_prefix}/status"
        
        if self.connected and self.client:
            self.client.publish(topic, payload, qos=1, retain=True)
    
    def _publish_or_queue(self, topic: str, payload: str) -> bool:
        """
        尝试直接发送，失败则入队
        """
        if self.connected and self.client:
            try:
                result = self.client.publish(topic, payload, qos=self.qos)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
            except Exception as e:
                logger.warning(f"Publish failed: {e}")
        
        # 入队缓存
        try:
            self.offline_queue.put({
                "topic": topic,
                "payload": payload,
                "timestamp": time.time()
            }, block=False)
            logger.debug(f"Message queued, queue size: {self.offline_queue.qsize()}")
            return True
        except Full:
            logger.error("Offline queue full, message dropped")
            return False
    
    def _start_flush_thread(self):
        """启动离线队列 flush 线程"""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def _flush_loop(self):
        """定期尝试发送队列中的离线消息"""
        while self._running:
            if self.connected and not self.offline_queue.empty():
                try:
                    msg = self.offline_queue.get(timeout=1)
                    result = self.client.publish(
                        msg["topic"], msg["payload"], qos=self.qos
                    )
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        logger.debug("Queued message flushed")
                    else:
                        # 发送失败，放回队列
                        self.offline_queue.put(msg)
                except Exception as e:
                    logger.debug(f"Flush error: {e}")
            
            time.sleep(self.reconnect_interval)
    
    def get_queue_stats(self) -> Dict:
        """获取队列统计"""
        return {
            "queue_size": self.offline_queue.qsize(),
            "connected": self.connected,
            "broker": f"{self.broker_host}:{self.broker_port}"
        }
