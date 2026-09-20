"""
Excel & CSV Export Utility for Network Security Decisions

Exports node decisions with explanations to Excel and CSV formats.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import os

try:
    from openpyxl.styles import Font, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class ExcelExporter:
    """Export node decisions and explanations to Excel and CSV."""
    
    def __init__(self, output_dir: str = "results_excel"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save Excel and CSV files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_decisions_to_excel(self, explanations: List[Dict[str, Any]], 
                                   episode: int, step: int = None) -> str:
        """
        Export node decisions to Excel.
        
        Args:
            explanations: List of explanation dictionaries from XAI engine
            episode: Episode number
            step: Step number (optional)
        
        Returns:
            Path to saved Excel file or None if openpyxl not available
        """
        if not HAS_OPENPYXL:
            print("[!] openpyxl not installed. Install with: pip install openpyxl")
            print("[!] Skipping Excel export. CSV export is available.")
            return None
        
        # Prepare data for DataFrame
        data = []
        
        for exp in explanations:
            node_id = exp['node_id']
            features = exp['features']
            
            row = {
                'Node ID': node_id,
                'Action': exp['action_name'],
                'Confidence (%)': exp['confidence_pct'],
                'Severity': exp['severity_level'],
                'Compromised': features['compromised'],
                'Risk Score': f"{features['risk_score']:.3f}",
                'Isolation Status': features['isolation_status'],
                'Vulnerability Score': f"{features['vulnerability_score']:.3f}",
                'Flow Duration': f"{features['flow_duration']:.3f}",
                'Packet Rate': f"{features['packet_rate']:.3f}",
                'Byte Rate': f"{features['byte_rate']:.3f}",
                'Node Degree': f"{features['node_degree']:.3f}",
                'GCN Risk': f"{exp['gcn_risk']:.3f}" if exp['gcn_risk'] is not None else "N/A",
                'Explanation': exp['action_explanation'],
                'Top Factor 1': f"{exp['top_contributing_features'][0]['feature_name']} ({exp['top_contributing_features'][0]['feature_value']:.2f})" if len(exp['top_contributing_features']) > 0 else "N/A",
                'Top Factor 2': f"{exp['top_contributing_features'][1]['feature_name']} ({exp['top_contributing_features'][1]['feature_value']:.2f})" if len(exp['top_contributing_features']) > 1 else "N/A",
                'Top Factor 3': f"{exp['top_contributing_features'][2]['feature_name']} ({exp['top_contributing_features'][2]['feature_value']:.2f})" if len(exp['top_contributing_features']) > 2 else "N/A",
            }
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if step is not None:
            filename = f"episode_{episode:03d}_step_{step:03d}_{timestamp}.xlsx"
        else:
            filename = f"episode_{episode:03d}_{timestamp}.xlsx"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to Excel with formatting
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Decisions', index=False)
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Decisions']
                
                # Auto-adjust column widths
                for idx, col in enumerate(df.columns, 1):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(col)
                    )
                    worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 50)
                
                # Add summary sheet
                summary_data = {
                    'Metric': [
                        'Total Nodes',
                        'Episode',
                        'Step' if step is not None else 'N/A',
                        'Timestamp',
                        'Average Confidence',
                        'Compromised Nodes',
                        'High Risk Nodes',
                        'Isolated Nodes'
                    ],
                    'Value': [
                        len(explanations),
                        episode,
                        step if step is not None else 'N/A',
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{np.mean([float(exp['confidence']) for exp in explanations]):.2%}",
                        sum(1 for exp in explanations if exp['features']['compromised'] == 1),
                        sum(1 for exp in explanations if exp['features']['risk_score'] > 0.7),
                        sum(1 for exp in explanations if exp['features']['isolation_status'] == 1)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                summary_sheet.column_dimensions['A'].width = 25
                summary_sheet.column_dimensions['B'].width = 25
            
            print(f"[+] Excel file saved: {filepath}")
        
        except Exception as e:
            print(f"[!] Error saving Excel file: {e}")
            return None
        
        return filepath
    
    def export_decisions_to_csv(self, explanations: List[Dict[str, Any]], 
                                episode: int, step: int = None) -> str:
        """
        Export node decisions to CSV.
        
        Args:
            explanations: List of explanation dictionaries from XAI engine
            episode: Episode number
            step: Step number (optional)
        
        Returns:
            Path to saved CSV file
        """
        # Prepare data for DataFrame
        data = []
        
        for exp in explanations:
            node_id = exp['node_id']
            features = exp['features']
            
            row = {
                'Node ID': node_id,
                'Action': exp['action_name'],
                'Confidence (%)': exp['confidence_pct'],
                'Severity': exp['severity_level'],
                'Compromised': features['compromised'],
                'Risk Score': f"{features['risk_score']:.3f}",
                'Isolation Status': features['isolation_status'],
                'Vulnerability Score': f"{features['vulnerability_score']:.3f}",
                'Flow Duration': f"{features['flow_duration']:.3f}",
                'Packet Rate': f"{features['packet_rate']:.3f}",
                'Byte Rate': f"{features['byte_rate']:.3f}",
                'Node Degree': f"{features['node_degree']:.3f}",
                'GCN Risk': f"{exp['gcn_risk']:.3f}" if exp['gcn_risk'] is not None else "N/A",
                'Explanation': exp['action_explanation'],
                'Top Factor 1': f"{exp['top_contributing_features'][0]['feature_name']} ({exp['top_contributing_features'][0]['feature_value']:.2f})" if len(exp['top_contributing_features']) > 0 else "N/A",
                'Top Factor 2': f"{exp['top_contributing_features'][1]['feature_name']} ({exp['top_contributing_features'][1]['feature_value']:.2f})" if len(exp['top_contributing_features']) > 1 else "N/A",
                'Top Factor 3': f"{exp['top_contributing_features'][2]['feature_name']} ({exp['top_contributing_features'][2]['feature_value']:.2f})" if len(exp['top_contributing_features']) > 2 else "N/A",
            }
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if step is not None:
            filename = f"episode_{episode:03d}_step_{step:03d}_{timestamp}.csv"
        else:
            filename = f"episode_{episode:03d}_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        try:
            df.to_csv(filepath, index=False)
            print(f"[+] CSV file saved: {filepath}")
        except Exception as e:
            print(f"[!] Error saving CSV file: {e}")
            return None
        
        return filepath
    
    def export_decisions_both(self, explanations: List[Dict[str, Any]], 
                             episode: int, step: int = None) -> tuple:
        """
        Export decisions to both Excel and CSV.
        
        Args:
            explanations: List of explanation dictionaries
            episode: Episode number
            step: Step number (optional)
        
        Returns:
            Tuple of (excel_path, csv_path)
        """
        excel_path = self.export_decisions_to_excel(explanations, episode, step)
        csv_path = self.export_decisions_to_csv(explanations, episode, step)
        return excel_path, csv_path
