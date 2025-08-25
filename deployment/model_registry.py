"""
Model Registry for AI Tutorial - Version Management and Metadata

This module provides a simple model registry system for managing
model versions, metadata, and deployment history.
"""

import os
import json
import datetime
import hashlib
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging


class ModelRegistry:
    """
    Simple model registry for version management and metadata tracking
    """
    
    def __init__(self, registry_dir: str = 'model_registry'):
        """
        Initialize model registry
        
        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.registry_dir / 'models'
        self.metadata_dir = self.registry_dir / 'metadata'
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Registry index file
        self.index_file = self.registry_dir / 'registry_index.json'
        
        # Load existing registry
        self.registry = self._load_registry()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ModelRegistry')
    
    def _load_registry(self) -> Dict:
        """Load registry index from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load registry index: {e}")
        
        return {
            'models': {},
            'created_at': datetime.datetime.now().isoformat(),
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save registry index to file"""
        self.registry['last_updated'] = datetime.datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(self, 
                      model_path: str, 
                      model_name: str, 
                      version: str = None,
                      description: str = "",
                      tags: List[str] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Register a new model or version
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            version: Version string (auto-generated if not provided)
            description: Model description
            tags: List of tags for categorization
            metadata: Additional metadata
        
        Returns:
            Model version ID
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Auto-generate version if not provided
        if version is None:
            existing_versions = self.list_versions(model_name)
            version = f"v{len(existing_versions) + 1}"
        
        # Calculate file hash for integrity checking
        file_hash = self._calculate_file_hash(model_path)
        
        # Create model entry if it doesn't exist
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {
                'created_at': datetime.datetime.now().isoformat(),
                'versions': {},
                'latest_version': version
            }
        
        # Check if this version already exists
        version_id = f"{model_name}:{version}"
        if version in self.registry['models'][model_name]['versions']:
            raise ValueError(f"Version {version} already exists for model {model_name}")
        
        # Copy model file to registry
        model_filename = f"{model_name}_{version}{Path(model_path).suffix}"
        registry_model_path = self.models_dir / model_filename
        shutil.copy2(model_path, registry_model_path)
        
        # Create metadata
        model_metadata = {
            'version': version,
            'description': description,
            'tags': tags or [],
            'file_hash': file_hash,
            'file_size': os.path.getsize(model_path),
            'original_path': model_path,
            'registry_path': str(registry_model_path),
            'registered_at': datetime.datetime.now().isoformat(),
            'model_type': self._detect_model_type(model_path),
            'custom_metadata': metadata or {}
        }
        
        # Add version to registry
        self.registry['models'][model_name]['versions'][version] = model_metadata
        self.registry['models'][model_name]['latest_version'] = version
        
        # Save metadata file
        metadata_file = self.metadata_dir / f"{model_name}_{version}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Registered model {version_id}")
        return version_id
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect model type based on file extension"""
        suffix = Path(model_path).suffix.lower()
        if suffix in ['.pkl', '.pickle']:
            return 'sklearn'
        elif suffix in ['.pth', '.pt']:
            return 'pytorch'
        elif suffix in ['.h5', '.hdf5']:
            return 'tensorflow'
        elif suffix == '.joblib':
            return 'joblib'
        else:
            return 'unknown'
    
    def get_model(self, model_name: str, version: str = None) -> Dict:
        """
        Get model information
        
        Args:
            model_name: Name of the model
            version: Version (latest if not specified)
        
        Returns:
            Model metadata dictionary
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.registry['models'][model_name]
        
        if version is None:
            version = model_info['latest_version']
        
        if version not in model_info['versions']:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        return model_info['versions'][version]
    
    def get_model_path(self, model_name: str, version: str = None) -> str:
        """
        Get path to model file in registry
        
        Args:
            model_name: Name of the model
            version: Version (latest if not specified)
        
        Returns:
            Path to model file
        """
        model_metadata = self.get_model(model_name, version)
        return model_metadata['registry_path']
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.registry['models'].keys())
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model
        
        Args:
            model_name: Name of the model
        
        Returns:
            List of version strings
        """
        if model_name not in self.registry['models']:
            return []
        
        return list(self.registry['models'][model_name]['versions'].keys())
    
    def delete_model(self, model_name: str, version: str = None):
        """
        Delete a model or specific version
        
        Args:
            model_name: Name of the model
            version: Version to delete (all versions if not specified)
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version is None:
            # Delete entire model
            model_info = self.registry['models'][model_name]
            
            # Delete all version files
            for ver, metadata in model_info['versions'].items():
                registry_path = Path(metadata['registry_path'])
                if registry_path.exists():
                    registry_path.unlink()
                
                # Delete metadata file
                metadata_file = self.metadata_dir / f"{model_name}_{ver}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
            
            # Remove from registry
            del self.registry['models'][model_name]
            self.logger.info(f"Deleted model {model_name} (all versions)")
        
        else:
            # Delete specific version
            if version not in self.registry['models'][model_name]['versions']:
                raise ValueError(f"Version {version} not found for model {model_name}")
            
            metadata = self.registry['models'][model_name]['versions'][version]
            
            # Delete model file
            registry_path = Path(metadata['registry_path'])
            if registry_path.exists():
                registry_path.unlink()
            
            # Delete metadata file
            metadata_file = self.metadata_dir / f"{model_name}_{version}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from registry
            del self.registry['models'][model_name]['versions'][version]
            
            # Update latest version if needed
            remaining_versions = list(self.registry['models'][model_name]['versions'].keys())
            if remaining_versions:
                self.registry['models'][model_name]['latest_version'] = remaining_versions[-1]
            else:
                # No versions left, delete the model entry
                del self.registry['models'][model_name]
            
            self.logger.info(f"Deleted model {model_name}:{version}")
        
        self._save_registry()
    
    def search_models(self, 
                     name_pattern: str = None,
                     tags: List[str] = None,
                     model_type: str = None) -> List[Dict]:
        """
        Search for models based on criteria
        
        Args:
            name_pattern: Pattern to match model names
            tags: Tags to filter by
            model_type: Model type to filter by
        
        Returns:
            List of matching model metadata
        """
        results = []
        
        for model_name, model_info in self.registry['models'].items():
            # Check name pattern
            if name_pattern and name_pattern.lower() not in model_name.lower():
                continue
            
            # Check each version
            for version, metadata in model_info['versions'].items():
                match = True
                
                # Check tags
                if tags:
                    model_tags = set(metadata.get('tags', []))
                    if not set(tags).issubset(model_tags):
                        match = False
                
                # Check model type
                if model_type and metadata.get('model_type') != model_type:
                    match = False
                
                if match:
                    results.append({
                        'model_name': model_name,
                        'version': version,
                        'model_id': f"{model_name}:{version}",
                        **metadata
                    })
        
        return results
    
    def get_registry_stats(self) -> Dict:
        """Get registry statistics"""
        total_models = len(self.registry['models'])
        total_versions = sum(len(info['versions']) for info in self.registry['models'].values())
        
        # Calculate total size
        total_size = 0
        model_types = {}
        
        for model_info in self.registry['models'].values():
            for metadata in model_info['versions'].values():
                total_size += metadata.get('file_size', 0)
                model_type = metadata.get('model_type', 'unknown')
                model_types[model_type] = model_types.get(model_type, 0) + 1
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'model_types': model_types,
            'registry_created': self.registry.get('created_at'),
            'last_updated': self.registry.get('last_updated')
        }
    
    def export_registry(self, export_path: str):
        """
        Export entire registry to a directory
        
        Args:
            export_path: Path to export directory
        """
        export_dir = Path(export_path)
        export_dir.mkdir(exist_ok=True)
        
        # Copy registry structure
        shutil.copytree(self.registry_dir, export_dir / 'model_registry', dirs_exist_ok=True)
        
        self.logger.info(f"Registry exported to {export_path}")
    
    def import_registry(self, import_path: str):
        """
        Import registry from a directory
        
        Args:
            import_path: Path to import directory
        """
        import_dir = Path(import_path) / 'model_registry'
        
        if not import_dir.exists():
            raise FileNotFoundError(f"Registry directory not found: {import_dir}")
        
        # Backup current registry
        backup_dir = self.registry_dir.parent / f"{self.registry_dir.name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(self.registry_dir, backup_dir)
        
        try:
            # Import new registry
            shutil.copytree(import_dir, self.registry_dir)
            
            # Reload registry
            self.registry = self._load_registry()
            
            self.logger.info(f"Registry imported from {import_path}")
            self.logger.info(f"Previous registry backed up to {backup_dir}")
        
        except Exception as e:
            # Restore backup on error
            shutil.move(backup_dir, self.registry_dir)
            raise e


def demonstrate_model_registry():
    """Demonstrate model registry functionality"""
    print("🗂️ Model Registry Demonstration")
    print("=" * 50)
    
    # Create registry
    registry = ModelRegistry('demo_model_registry')
    
    # Check if we have some models to register
    model_files = ['simple_model.pth', 'complete_model.pth']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model_name = model_file.replace('.pth', '').replace('_model', '')
                version_id = registry.register_model(
                    model_path=model_file,
                    model_name=model_name,
                    description=f"Demo {model_name} model",
                    tags=['demo', 'neural_network', 'tutorial'],
                    metadata={'framework': 'pytorch', 'accuracy': 0.85}
                )
                print(f"✅ Registered {version_id}")
            except Exception as e:
                print(f"⚠️ Could not register {model_file}: {e}")
    
    # Show registry stats
    print(f"\n📊 Registry Statistics:")
    stats = registry.get_registry_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # List all models
    print(f"\n📋 Registered Models:")
    for model_name in registry.list_models():
        versions = registry.list_versions(model_name)
        print(f"  📁 {model_name}: {', '.join(versions)}")
    
    # Search example
    print(f"\n🔍 Search Results (neural network models):")
    results = registry.search_models(tags=['neural_network'])
    for result in results:
        print(f"  🎯 {result['model_id']}: {result['description']}")
    
    print(f"\n✅ Model registry demonstration completed!")
    print(f"📁 Registry stored in: {registry.registry_dir}")


if __name__ == "__main__":
    demonstrate_model_registry()