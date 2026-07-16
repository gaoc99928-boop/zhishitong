"""
本地数据存储模块
SQLite 持久化 + 7 天滚动清理
支持结构化查询和 CSV 导出
"""
import sqlite3
import json
import logging
import time
import csv
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EdgeDataStore:
    """
    边缘数据存储器
    
    功能:
    - 结构化存储识别结果
    - 按时间范围查询
    - 7 天自动滚动清理
    - CSV 导出
    - 车牌频次统计
    """
    
    def __init__(
        self,
        db_path: str = "./data/records.db",
        retention_days: int = 7,
        max_records: int = 100000
    ):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.max_records = max_records
        
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接上下文"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """初始化数据表"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    frame_id INTEGER,
                    track_id INTEGER,
                    plate_number TEXT,
                    plate_color TEXT,
                    plate_conf REAL,
                    vehicle_type TEXT,
                    detection_conf REAL,
                    scene_type TEXT,
                    scene_enhanced INTEGER DEFAULT 0,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    processing_time_ms REAL,
                    node_id TEXT,
                    image_path TEXT,
                    metadata TEXT
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON vehicle_records(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plate 
                ON vehicle_records(plate_number)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scene 
                ON vehicle_records(scene_type)
            """)
            
            conn.commit()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def insert(self, result: Dict, node_id: str = "edge-01") -> int:
        """
        插入识别结果
        
        Args:
            result: 流水线输出结果
            node_id: 边缘节点标识
            
        Returns:
            插入的记录 ID
        """
        ts = time.time()
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        
        with self._get_connection() as conn:
            for vehicle in result.get("vehicles", []):
                plate = vehicle.get("plate", {}) or {}
                bbox = vehicle.get("bbox", [0, 0, 0, 0])
                
                conn.execute("""
                    INSERT INTO vehicle_records (
                        timestamp, datetime, frame_id, track_id,
                        plate_number, plate_color, plate_conf,
                        vehicle_type, detection_conf,
                        scene_type, scene_enhanced,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        processing_time_ms, node_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ts, dt,
                    result.get("frame_id", 0),
                    vehicle.get("track_id", 0),
                    plate.get("number", ""),
                    plate.get("color", ""),
                    plate.get("confidence", 0.0),
                    vehicle.get("vehicle_type", ""),
                    vehicle.get("detection_conf", 0.0),
                    result.get("scene_type", "normal"),
                    1 if vehicle.get("scene_enhanced") else 0,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    result.get("processing_time_ms", 0.0),
                    node_id,
                    json.dumps(plate, ensure_ascii=False)
                ))
            
            conn.commit()
            
            # 检查并清理旧数据
            self._cleanup_old_records(conn)
            
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        plate_number: Optional[str] = None,
        scene_type: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        条件查询
        
        Args:
            start_time: 起始时间戳
            end_time: 结束时间戳
            plate_number: 车牌号模糊匹配
            scene_type: 场景类型
            vehicle_type: 车型
            limit: 返回数量限制
            offset: 分页偏移
        """
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if plate_number:
            conditions.append("plate_number LIKE ?")
            params.append(f"%{plate_number}%")
        if scene_type:
            conditions.append("scene_type = ?")
            params.append(scene_type)
        if vehicle_type:
            conditions.append("vehicle_type = ?")
            params.append(vehicle_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT * FROM vehicle_records
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_stats(self, hours: int = 24) -> Dict:
        """
        获取统计信息
        
        Args:
            hours: 过去 N 小时的统计
        """
        since = time.time() - hours * 3600
        
        with self._get_connection() as conn:
            # 总过车数
            total = conn.execute(
                "SELECT COUNT(*) FROM vehicle_records WHERE timestamp >= ?",
                (since,)
            ).fetchone()[0]
            
            # 场景分布
            scenes = conn.execute("""
                SELECT scene_type, COUNT(*) as count
                FROM vehicle_records
                WHERE timestamp >= ?
                GROUP BY scene_type
            """, (since,)).fetchall()
            
            # 车型分布
            types = conn.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM vehicle_records
                WHERE timestamp >= ?
                GROUP BY vehicle_type
            """, (since,)).fetchall()
            
            # 平均处理延迟
            latency = conn.execute("""
                SELECT AVG(processing_time_ms) FROM vehicle_records
                WHERE timestamp >= ?
            """, (since,)).fetchone()[0]
            
            return {
                "total_records": total,
                "scene_distribution": {row["scene_type"]: row["count"] for row in scenes},
                "vehicle_type_distribution": {row["vehicle_type"]: row["count"] for row in types},
                "avg_latency_ms": round(latency or 0, 2),
                "period_hours": hours
            }
    
    def export_csv(self, filepath: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """
        导出查询结果为 CSV
        """
        records = self.query(start_time=start_time, end_time=end_time, limit=100000)
        
        if not records:
            logger.warning("No records to export")
            return
        
        fieldnames = [
            "datetime", "plate_number", "plate_color", "plate_conf",
            "vehicle_type", "detection_conf", "scene_type",
            "processing_time_ms", "node_id"
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({k: record.get(k, "") for k in fieldnames})
        
        logger.info(f"Exported {len(records)} records to {filepath}")
    
    def _cleanup_old_records(self, conn: sqlite3.Connection):
        """清理过期数据"""
        cutoff = time.time() - self.retention_days * 86400
        
        # 先检查总数
        count = conn.execute("SELECT COUNT(*) FROM vehicle_records").fetchone()[0]
        
        if count > self.max_records:
            # 保留最新的 max_records 条
            conn.execute("""
                DELETE FROM vehicle_records
                WHERE id NOT IN (
                    SELECT id FROM vehicle_records
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """, (self.max_records,))
        
        # 清理过期数据
        conn.execute("DELETE FROM vehicle_records WHERE timestamp < ?", (cutoff,))
        conn.commit()
