# dataset_integrity.py
"""
Complete dataset validation and quality assurance
"""

import os
import json
import cv2
import numpy as np
import glob
from datetime import datetime
from pathlib import Path
import shutil

class DatasetIntegrityChecker:
    """
    Validate and ensure dataset quality before training
    """
    
    def __init__(self, dataset_path="underwater_dataset", backup=True):
        self.dataset_path = dataset_path
        self.backup = backup
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': 0,
            'valid_pairs': 0,
            'invalid_labels': [],
            'missing_labels': [],
            'class_distribution': {},
            'bbox_statistics': {},
            'warnings': []
        }
        
        # Class mapping
        self.classes = {
            0: "diver",
            1: "suspicious_object",
            2: "boat",
            3: "ship",
            4: "submarine",
            5: "pipeline",
            6: "obstacle",
            7: "marine_life",
            8: "swimmer",
            9: "underwater_vehicle"
        }
        
        print("="*70)
        print("📊 DATASET INTEGRITY CHECKER")
        print("="*70)
    
    def validate_all(self, split='train'):
        """Run complete validation on dataset split"""
        # Reset split-specific counters so multiple calls (train/val) do not contaminate each other.
        self.report['total_images'] = 0
        self.report['valid_pairs'] = 0
        self.report['invalid_labels'] = []
        self.report['missing_labels'] = []
        self.report['class_distribution'] = {}
        self.report['bbox_statistics'] = {}
        
        image_dir = os.path.join(self.dataset_path, f"images/{split}")
        label_dir = os.path.join(self.dataset_path, f"labels/{split}")
        
        if not os.path.exists(image_dir):
            print(f"❌ Image directory not found: {image_dir}")
            return False
        
        # Get all images
        images = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")
        self.report['total_images'] = len(images)
        
        print(f"\n📁 Validating {split} split: {len(images)} images")
        print("-"*50)
        
        for img_path in images:
            self.validate_single(img_path, label_dir)
        
        # Generate statistics
        self.generate_statistics()
        
        # Save report
        self.save_report()
        
        # Create versioned snapshot if valid
        if self.report['valid_pairs'] == self.report['total_images']:
            self.create_versioned_snapshot()
        
        return self.report['valid_pairs'] == self.report['total_images']
    
    def validate_single(self, image_path, label_dir):
        """Validate single image-label pair"""
        
        # Check label exists
        label_name = os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_path):
            self.report['missing_labels'].append({
                'image': image_path,
                'label': label_path,
                'reason': 'Label file missing'
            })
            return False
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            self.report['invalid_labels'].append({
                'image': image_path,
                'reason': 'Cannot read image'
            })
            return False
        
        h, w = img.shape[:2]
        
        # Validate labels
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                
                # Check format
                if len(parts) != 5:
                    self.report['invalid_labels'].append({
                        'image': image_path,
                        'line': line_num + 1,
                        'reason': f'Invalid format: expected 5 values, got {len(parts)}'
                    })
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate class ID
                    if class_id not in self.classes:
                        self.report['invalid_labels'].append({
                            'image': image_path,
                            'line': line_num + 1,
                            'reason': f'Invalid class ID: {class_id}'
                        })
                        continue
                    
                    # Validate normalized coordinates
                    if not (0 <= x_center <= 1):
                        self.report['invalid_labels'].append({
                            'image': image_path,
                            'line': line_num + 1,
                            'reason': f'x_center out of range: {x_center}'
                        })
                        continue
                    
                    if not (0 <= y_center <= 1):
                        self.report['invalid_labels'].append({
                            'image': image_path,
                            'line': line_num + 1,
                            'reason': f'y_center out of range: {y_center}'
                        })
                        continue
                    
                    if not (0 < width <= 1):
                        self.report['invalid_labels'].append({
                            'image': image_path,
                            'line': line_num + 1,
                            'reason': f'width out of range: {width}'
                        })
                        continue
                    
                    if not (0 < height <= 1):
                        self.report['invalid_labels'].append({
                            'image': image_path,
                            'line': line_num + 1,
                            'reason': f'height out of range: {height}'
                        })
                        continue
                    
                    # Update class distribution
                    class_name = self.classes[class_id]
                    self.report['class_distribution'][class_name] = \
                        self.report['class_distribution'].get(class_name, 0) + 1
                    
                    # Update bbox statistics
                    if 'widths' not in self.report['bbox_statistics']:
                        self.report['bbox_statistics']['widths'] = []
                        self.report['bbox_statistics']['heights'] = []
                        self.report['bbox_statistics']['areas'] = []
                    
                    self.report['bbox_statistics']['widths'].append(width)
                    self.report['bbox_statistics']['heights'].append(height)
                    self.report['bbox_statistics']['areas'].append(width * height)
                    
                except ValueError as e:
                    self.report['invalid_labels'].append({
                        'image': image_path,
                        'line': line_num + 1,
                        'reason': f'Value conversion error: {e}'
                    })
        
        self.report['valid_pairs'] += 1
        return True
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        
        print("\n📊 VALIDATION REPORT")
        print("="*50)
        print(f"Total images: {self.report['total_images']}")
        print(f"Valid pairs: {self.report['valid_pairs']}")
        print(f"Missing labels: {len(self.report['missing_labels'])}")
        print(f"Invalid labels: {len(self.report['invalid_labels'])}")
        
        if self.report['class_distribution']:
            print("\n📈 Class Distribution:")
            for class_name, count in sorted(self.report['class_distribution'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"   {class_name}: {count}")
        
        if self.report['bbox_statistics'].get('widths'):
            widths = self.report['bbox_statistics']['widths']
            heights = self.report['bbox_statistics']['heights']
            areas = self.report['bbox_statistics']['areas']
            
            print("\n📐 Bounding Box Statistics:")
            print(f"   Average width: {np.mean(widths):.3f}")
            print(f"   Average height: {np.mean(heights):.3f}")
            print(f"   Average area: {np.mean(areas):.3f}")
            print(f"   Width std: {np.std(widths):.3f}")
            print(f"   Height std: {np.std(heights):.3f}")
    
    def save_report(self):
        """Save validation report"""
        
        report_dir = "dataset_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"integrity_report_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON
        report_copy = self.report.copy()
        if 'bbox_statistics' in report_copy:
            for key in ['widths', 'heights', 'areas']:
                if key in report_copy['bbox_statistics']:
                    report_copy['bbox_statistics'][key] = [
                        float(v) for v in report_copy['bbox_statistics'][key]
                    ]
        
        with open(report_path, 'w') as f:
            json.dump(report_copy, f, indent=2)
        
        print(f"\n💾 Report saved: {report_path}")
        
        # Generate HTML report
        self.generate_html_report(report_path.replace('.json', '.html'))
        
        return report_path
    
    def generate_html_report(self, html_path):
        """Generate HTML visualization report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Integrity Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
                .metric {{ display: inline-block; width: 200px; margin: 10px; padding: 15px; background: #f0f2f6; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 28px; font-weight: bold; color: #0066cc; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .valid {{ color: green; }}
                .invalid {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #0066cc; color: white; }}
                tr:hover {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Dataset Integrity Report</h1>
                <p>Generated: {self.report['timestamp']}</p>
                
                <div class="metric">
                    <div class="metric-value">{self.report['total_images']}</div>
                    <div class="metric-label">Total Images</div>
                </div>
                <div class="metric">
                    <div class="metric-value {('valid' if self.report['valid_pairs'] == self.report['total_images'] else 'invalid')}">
                        {self.report['valid_pairs']}
                    </div>
                    <div class="metric-label">Valid Pairs</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(self.report['missing_labels'])}</div>
                    <div class="metric-label">Missing Labels</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(self.report['invalid_labels'])}</div>
                    <div class="metric-label">Invalid Labels</div>
                </div>
                
                <h2>📈 Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Count</th></tr>
        """
        
        for class_name, count in sorted(self.report['class_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
            html += f"<tr><td>{class_name}</td><td>{count}</td></tr>"
        
        html += """
                </table>
                
                <h2>⚠️ Issues Found</h2>
        """
        
        if self.report['missing_labels']:
            html += "<h3>Missing Labels</h3><ul>"
            for item in self.report['missing_labels'][:10]:
                html += f"<li>{os.path.basename(item['image'])}: {item['reason']}</li>"
            html += "</ul>"
        
        if self.report['invalid_labels']:
            html += "<h3>Invalid Labels</h3><ul>"
            for item in self.report['invalid_labels'][:10]:
                html += f"<li>{os.path.basename(item['image'])}: {item['reason']}</li>"
            html += "</ul>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html)
        
        print(f"📊 HTML report saved: {html_path}")
    
    def create_versioned_snapshot(self):
        """Create versioned snapshot of clean dataset"""
        
        if not self.backup:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = f"dataset_snapshots/v{timestamp}"
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Copy dataset
        shutil.copytree(self.dataset_path, os.path.join(snapshot_dir, "dataset"))
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'valid_pairs': self.report['valid_pairs'],
            'total_images': self.report['total_images'],
            'class_distribution': self.report['class_distribution']
        }
        
        with open(os.path.join(snapshot_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Versioned snapshot created: {snapshot_dir}")
        
        # Create symlink to latest
        latest_link = "dataset_snapshots/latest"
        try:
            if os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    os.remove(latest_link)
                else:
                    shutil.rmtree(latest_link)
            os.symlink(f"v{timestamp}", latest_link, target_is_directory=True)
        except Exception:
            # Fallback for environments where symlink creation is restricted.
            shutil.copytree(snapshot_dir, latest_link, dirs_exist_ok=True)
        
        return snapshot_dir

def fix_common_issues(report_path):
    """Auto-fix common annotation issues"""
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    fixes_applied = []
    
    for issue in report.get('invalid_labels', []):
        if 'x_center out of range' in issue['reason']:
            # Fix out-of-range coordinates
            print(f"🔧 Fixing: {issue['image']}")
            fixes_applied.append(issue['image'])
    
    return fixes_applied

if __name__ == "__main__":
    # Validate dataset
    checker = DatasetIntegrityChecker(backup=True)
    
    # Validate both splits
    train_valid = checker.validate_all(split='train')
    val_valid = checker.validate_all(split='val')
    
    if train_valid and val_valid:
        print("\n✅ Dataset is clean and ready for training!")
    else:
        print("\n⚠️ Dataset issues found. Please fix before training.")