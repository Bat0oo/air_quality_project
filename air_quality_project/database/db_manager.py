import pymongo
from pymongo import MongoClient
import config
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = MongoClient(config.MONGO_URI)
        self.db = self.client[config.MONGO_DB]
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        # Raw data indexes
        self.db[config.COLLECTION_RAW].create_index([("city", 1), ("timestamp", -1)])
        self.db[config.COLLECTION_RAW].create_index([("parameter", 1)])
        
        # Processed data indexes
        self.db[config.COLLECTION_PROCESSED].create_index([("city", 1), ("date", -1)])
        
    def insert_many(self, collection_name, documents):
        """Insert multiple documents"""
        if not documents:
            return 0
        try:
            result = self.db[collection_name].insert_many(documents, ordered=False)
            return len(result.inserted_ids)
        except pymongo.errors.BulkWriteError as e:
            # Return count excluding duplicates
            return len(documents) - len(e.details.get('writeErrors', []))
    
    def insert_one(self, collection_name, document):
        """Insert single document"""
        result = self.db[collection_name].insert_one(document)
        return result.inserted_id
    
    def find(self, collection_name, query=None, projection=None):
        """Find documents"""
        if query is None:
            query = {}
        return list(self.db[collection_name].find(query, projection))
    
    def find_one(self, collection_name, query):
        """Find single document"""
        return self.db[collection_name].find_one(query)
    
    def update_one(self, collection_name, query, update):
        """Update single document"""
        return self.db[collection_name].update_one(query, update)
    
    def delete_many(self, collection_name, query):
        """Delete multiple documents"""
        return self.db[collection_name].delete_many(query)
    
    def count(self, collection_name, query=None):
        """Count documents"""
        if query is None:
            query = {}
        return self.db[collection_name].count_documents(query)
    
    def aggregate(self, collection_name, pipeline):
        """Run aggregation pipeline"""
        return list(self.db[collection_name].aggregate(pipeline))
    
    def get_stats(self):
        """Get database statistics"""
        return {
            'raw': self.count(config.COLLECTION_RAW),
            'processed': self.count(config.COLLECTION_PROCESSED),
            'daily': self.count(config.COLLECTION_DAILY),
            'hourly': self.count(config.COLLECTION_HOURLY),
            'predictions': self.count(config.COLLECTION_PREDICTIONS)
        }
    
    def clear_collection(self, collection_name):
        """Clear all documents from a collection"""
        return self.db[collection_name].delete_many({})
    
    def close(self):
        """Close database connection"""
        self.client.close()

if __name__ == "__main__":
    # Test connection
    db = DatabaseManager()
    print("✅ Database connection successful")
    stats = db.get_stats()
    print(f"📊 Database stats: {stats}")
    db.close()