import argparse
import os
import time
from typing import List, Dict, Optional, Union
import logging

import requests
from dotenv import load_dotenv

from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import pandas as pd
import json

load_dotenv()

logger = logging.getLogger(__name__)
server_host = os.getenv("SERVER_HOST")
port = os.getenv("PORT")

class VilaM3Client:
    def __init__(self, base_url: str = "http://0.0.0.0:8585", 
                openai_api_key: str = None,
                openai_endpoint: str = "https://api.openai.com/v1/chat/completions",
                openai_model: str = "gpt-4o-mini"):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.openai_endpoint = openai_endpoint
        self.openai_model = openai_model
        
    def send_single_image(self, image_path: str, prompt_text: str = None) -> Dict:
        """Send a single image for inference with optional user message"""
        logger.debug(f"image_path: {image_path}")

        with open(image_path, 'rb') as f:
            files = {'image_file': (Path(image_path).name, f)}
            data = {'prompt_text': prompt_text} if prompt_text else None
            try:
                response = requests.post(f"{self.base_url}/single", files=files, data=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Failed to process image {image_path}: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "filename": image_path
                }
    
    def send_batch_images(self, image_paths: List[str], prompt_text: str = None) -> Dict:
        """Send multiple images for batch inference with optional user message"""
        results = []
        temp_files = []
        
        try:
            # First prepare all files and store their content
            for path in image_paths:
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                        temp_files.append(('files', (str(Path(path).absolute()), content)))
                except Exception as e:
                    print(f"[ERROR] Failed to open {path}: {e}")
                    results.append({
                        "filename": path,
                        "status": "failed",
                        "error": str(e)
                    })
            
            if not temp_files:
                return {"status": "failed", "results": results}
                
            # Make the request with all file contents
            data = {'prompt_text': prompt_text} if prompt_text else None
            response = requests.post(f"{self.base_url}/batch_inference", files=temp_files, data=data)
            response.raise_for_status()
            batch_result = response.json()
            
            # Merge batch results with any individual failures
            if "results" in batch_result:
                batch_result["results"].extend(results)
            return batch_result
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Batch inference failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": results
            }
    
    def find_png_files(self, root_dir: str) -> List[str]:
        """Recursively find all PNG files in directory and subdirectories"""
        return glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)

    def load_ground_truth(self, excel_path: str) -> Dict[str, str]:
        """Load ground truth labels from Excel file and translate to English"""
        df = pd.read_excel(excel_path)
        ground_truth = {}
        translation = {
            '正常': 'Normal',
            '異常': 'Abnormal'
        }
        for _, row in df.iterrows():
            if 'SCHE_NO' in row and 'REP' in row:
                # Convert SCHE_NO to string and strip any whitespace
                sche_no = str(row['SCHE_NO']).strip()
                # Translate Chinese labels to English
                ground_truth[sche_no] = translation.get(row['REP'], row['REP'])
        print(f"[DEBUG] Loaded ground truth: {ground_truth}")
        return ground_truth

    def parse_ai_message(self, message: str) -> str:
        """Parse AI message to determine Normal/Abnormal classification"""
        print(f"\n[DEBUG] Parsing AI message: {message[:200]}...")  # Log first 200 chars
        
        # First check for explicit "Impression" section
        # if "### Conclusion" in message or "### Summary" in message or "IMPRESSION:" in message:
        #     impression_section = message.split("### Conclusion")[-1] if "### Conclusion" in message else message
        #     impression_section = impression_section.split("### Summary")[-1] if "### Summary" in message else impression_section
        #     impression_section = impression_section.split("IMPRESSION:")[-1] if "IMPRESSION:" in message else impression_section
            
        #     if "no acute" in impression_section.lower() or "normal" in impression_section.lower():
        #         print("[DEBUG] Found normal impression in conclusion")
        #         return "Normal"
        #     elif "abnormal" in impression_section.lower():
        #         print("[DEBUG] Found abnormal impression in conclusion")
        #         return "Abnormal"
        
        # Fallback to probability analysis
        if "Probability" in message or "probability" in message:
            high_prob_count = sum(1 for line in message.split('\n') 
                                if any(x in line.lower() for x in ['high likelihood', 'moderate likelihood'])
                                and float(line.split(':')[-1].strip().split()[0]) > 0.5)
            if high_prob_count > 1:
                print(f"[DEBUG] Found {high_prob_count} high probability pathologies")
                return "Abnormal"
            # else:
            #     print("[DEBUG] No significant pathologies found")
            #     return "Normal"
                
        # Final fallback to simple keyword matching
        # message_lower = message.lower()
        # if "normal" in message_lower and "abnormal" not in message_lower:
        #     print("[DEBUG] Keyword match: Normal")
        #     return "Normal"
        # elif "abnormal" in message_lower:
        #     print("[DEBUG] Keyword match: Abnormal")
        #     return "Abnormal"
            
        # print("[DEBUG] No clear classification found")
        # return "Unknown"
        
        # Use OpenAI API for more accurate classification
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        prompt = f"""
        Analyze this radiology report and classify it as either 'Normal', 'Abnormal' or 'Unknown':
        {message}
        
        Respond ONLY with either 'Normal', 'Abnormal' or 'Unknown', nothing else.
        """
        
        data = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.openai_endpoint, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Unknown"

    def aggregate_case_predictions(self, case_predictions: List[Dict], voting_threshold: float = 0.5) -> str:
        """
        Aggregate multiple image predictions for a single case
        
        Args:
            case_predictions: List of prediction dictionaries for images in the same case
            voting_threshold: Threshold for determining case label (default: 0.5)
            
        Returns:
            Final prediction label for the case ("Normal", "Abnormal", or "Unknown")
        """
        # Count predictions by label
        label_counts = {"Normal": 0, "Abnormal": 0, "Unknown": 0}
        
        for pred in case_predictions:
            try:
                ai_message = pred['result']['messages'][-1]['content']
                pred_label = self.parse_ai_message(ai_message)
                label_counts[pred_label] += 1
            except Exception as e:
                print(f"[ERROR] Failed to parse prediction: {e}")
                label_counts["Unknown"] += 1
        
        total_images = sum(label_counts.values())
        
        # If no valid predictions, return Unknown
        if total_images == 0:
            return "Unknown"
            
        # Calculate percentages
        normal_pct = label_counts["Normal"] / total_images
        abnormal_pct = label_counts["Abnormal"] / total_images
        
        # Log the voting results
        print(f"[DEBUG] Case voting results: Normal={label_counts['Normal']} ({normal_pct:.2f}), "
              f"Abnormal={label_counts['Abnormal']} ({abnormal_pct:.2f}), "
              f"Unknown={label_counts['Unknown']}")
        
        # Determine final label based on voting
        if abnormal_pct >= voting_threshold:
            return "Abnormal"
        elif normal_pct >= voting_threshold:
            return "Normal"
        else:
            return "Unknown"
    
    def calculate_confusion_matrix(self, 
                                 predictions: List[Dict], 
                                 ground_truth: Dict[str, str],
                                 labels: Optional[List[str]] = None,
                                 voting_threshold: float = 0.5) -> Dict:
        """Calculate and visualize confusion matrix with detailed Excel report"""
        print(f"\n[DEBUG] Calculating confusion matrix with {len(predictions)} predictions")
        print(f"[DEBUG] Ground truth keys: {list(ground_truth.keys())}")
        print(f"[DEBUG] Labels: {labels}")
        
        # Group predictions by SCHE_NO
        case_predictions = {}
        for pred in predictions:
            filename = pred['filename']
            full_path = Path(filename).absolute()
            
            # Extract SCHE_NO from path components
            sche_no = None
            for part in full_path.parts:
                # Check for all-digit SCHE_NO (original format)
                if part.isdigit() and len(part) >= 8:
                    sche_no = part
                    break
                # Check for P-prefixed SCHE_NO format (e.g., P1312110476)
                elif part.startswith('P') and part[1:].isdigit() and len(part) >= 9:
                    sche_no = part
                    break
            
            if not sche_no:
                print(f"[ERROR] Could not extract SCHE_NO from: {filename}")
                continue
                
            if sche_no not in case_predictions:
                case_predictions[sche_no] = []
            
            case_predictions[sche_no].append(pred)
        
        # Process each case to get aggregated predictions
        y_true = []
        y_pred = []
        case_report_data = []
        image_report_data = []
        
        for sche_no, preds in case_predictions.items():
            print(f"\n[DEBUG] Processing case {sche_no} with {len(preds)} images")
            
            # Skip cases not in ground truth
            if sche_no not in ground_truth:
                print(f"[ERROR] SCHE_NO {sche_no} not found in ground truth")
                print(f"[DEBUG] Available ground truth keys: {list(ground_truth.keys())}")
                
                # Add to image report as excluded
                for pred in preds:
                    image_report_data.append({
                        'File': pred['filename'],
                        'SCHE_NO': sche_no,
                        'Included_In_Matrix': 'No',
                        'Reason': 'SCHE_NO not found in ground truth',
                        'Image_Prediction': 'N/A',
                        'Case_Prediction': 'N/A',
                        'Ground_Truth': 'N/A'
                    })
                continue
            
            # Get ground truth for this case
            true_label = ground_truth[sche_no]
            
            # Process individual images
            image_predictions = []
            for pred in preds:
                try:
                    ai_message = pred['result']['messages'][-1]['content']
                    image_pred = self.parse_ai_message(ai_message)
                    
                    image_predictions.append({
                        'filename': pred['filename'],
                        'prediction': image_pred,
                        'ai_message': ai_message
                    })
                    
                    # Add to image report
                    image_report_data.append({
                        'File': pred['filename'],
                        'SCHE_NO': sche_no,
                        'Included_In_Matrix': 'Yes',
                        'Reason': 'Successfully processed',
                        'Image_Prediction': image_pred,
                        'Ground_Truth': true_label,
                        'AI_Message_Snippet': ai_message
                    })
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process image {pred['filename']}: {e}")
                    image_report_data.append({
                        'File': pred['filename'],
                        'SCHE_NO': sche_no,
                        'Included_In_Matrix': 'No',
                        'Reason': f'Processing error: {str(e)}',
                        'Image_Prediction': 'Error',
                        'Ground_Truth': true_label
                    })
            
            # Aggregate predictions for this case
            case_pred = self.aggregate_case_predictions(preds, voting_threshold)
            print(f"[DEBUG] Case {sche_no} final prediction: {case_pred} (Ground truth: {true_label})")
            
            # Add to confusion matrix data
            y_true.append(true_label)
            y_pred.append(case_pred)
            
            # Add to case report
            case_report_data.append({
                'SCHE_NO': sche_no,
                'Total_Images': len(preds),
                'Ground_Truth': true_label,
                'Case_Prediction': case_pred,
                'Normal_Count': sum(1 for img in image_predictions if img['prediction'] == 'Normal'),
                'Abnormal_Count': sum(1 for img in image_predictions if img['prediction'] == 'Abnormal'),
                'Unknown_Count': sum(1 for img in image_predictions if img['prediction'] == 'Unknown'),
                'Match_Ground_Truth': case_pred == true_label
            })
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Plot confusion matrix
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        #            xticklabels=labels, yticklabels=labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix (Case Level)')
        # plt.savefig('confusion_matrix.png')
        # plt.close()

        # Save detailed reports
        case_df = pd.DataFrame(case_report_data)
        image_df = pd.DataFrame(image_report_data)
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter('confusion_matrix_report.xlsx') as writer:
            case_df.to_excel(writer, sheet_name='Case_Level', index=False)
            image_df.to_excel(writer, sheet_name='Image_Level', index=False)
        
        # Print summary
        print(f"\n[REPORT] Total cases: {len(case_report_data)}")
        print(f"[REPORT] Total images: {len(image_report_data)}")
        print(f"[REPORT] Cases matching ground truth: {sum(1 for case in case_report_data if case['Match_Ground_Truth'])}")
        
        return {
            "matrix": cm.tolist(),
            "labels": labels,
            "plot_path": "confusion_matrix.png",
            "report_path": "confusion_matrix_report.xlsx",
            "total_cases": len(case_report_data),
            "total_images": len(image_report_data)
        }
    
    def health_check(self) -> Dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

def main():
    logfile = os.getenv("LOGFILE")
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in " "function %(funcName)s] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    parser = argparse.ArgumentParser(description='Vila M3 Client')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Single image parser
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('image_path', help='Path to image file')
    single_parser.add_argument('--prompt_text', help='Optional message to include with the image')
    
    # Batch images parser
    # batch_parser = subparsers.add_parser('batch')
    # batch_parser.add_argument('path', help='Path to image file or directory')
    # batch_parser.add_argument('--recursive', action='store_true', 
    #                         help='Recursively scan directory for PNG files')
    # batch_parser.add_argument('--prompt_text', help='Optional message to include with all images')
    # batch_parser.add_argument('--ground_truth_excel', 
    #                         help='Excel file containing ground truth labels')
    # batch_parser.add_argument('--labels', nargs='+',
    #                         help='Class labels for confusion matrix')
    # batch_parser.add_argument('--voting-threshold', type=float, default=0.5,
    #                         help='Threshold for case-level voting (default: 0.5)')
    # batch_parser.add_argument('--openai-api-key',
    #                         help='OpenAI API key for enhanced classification')
    # batch_parser.add_argument('--openai-endpoint',
    #                         help='OpenAI API endpoint',
    #                         default="https://api.openai.com/v1/chat/completions")
    # batch_parser.add_argument('--openai-model',
    #                         help='OpenAI model name',
    #                         default="gpt-4o-mini")
    
    # # Health check parser
    # health_parser = subparsers.add_parser('health')
    
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    
    base_url = f"{server_host}:{port}"
    # Only pass OpenAI params for batch command
    if args.command == 'single':
        client = VilaM3Client(base_url)
        result = client.send_single_image(args.image_path, args.prompt_text)
        logger.info(json.dumps(result, indent=2))

        predictions = []
        for res in result['choices']:
            if 'content' in res['message']['content']:
                inner_contents = res['message']['content']['content']
                logger.debug(f"inner_contents[0]: {inner_contents[0]}")
                predictions.append({
                    'text': inner_contents[0]['text']
                })
        logger.debug(f"single image result: {predictions}")
    elif args.command == 'batch':
        client = VilaM3Client(
            base_url,
            openai_api_key=args.openai_api_key,
            openai_endpoint=args.openai_endpoint,
            openai_model=args.openai_model
        )
        # Get image paths
        if os.path.isdir(args.path):
            if args.recursive:
                image_paths = client.find_png_files(args.path)
            else:
                image_paths = glob.glob(os.path.join(args.path, '*.png'))
        else:
            image_paths = [args.path]
            
        if not image_paths:
            print("Error: No PNG files found")
            return
        
        # No need to add voting threshold parameter here as it's already defined above
        
        # Process batch
        result = client.send_batch_images(image_paths, args.prompt_text)
        logger.info(json.dumps(result, indent=2))
        
        # Handle confusion matrix if requested
        if args.ground_truth_excel and args.labels:
            # Load ground truth from Excel
            ground_truth = client.load_ground_truth(args.ground_truth_excel)
            
            # Prepare predictions with filenames
            predictions = []
            for res in result['choices']:
                if 'content' in res['message']['content']:
                    inner_contents = res['message']['content']['content']
                    logger.debug(f"inner_contents[0]: {inner_contents[0]}")
                    predictions.append({
                        'text': inner_contents[0]['text']
                    })
            
            # Get voting threshold from args or use default
            voting_threshold = getattr(args, 'voting_threshold', 0.5)
            
            # Calculate confusion matrix
            cm_result = client.calculate_confusion_matrix(
                predictions, ground_truth, args.labels, voting_threshold
            )
            logger.debug("\nConfusion Matrix Results:")
            logger.debug(f"- Matrix plot: {cm_result['plot_path']}")
            logger.debug(f"- Detailed report: {cm_result['report_path']}")
            logger.debug(f"- Total cases: {cm_result['total_cases']}")
            logger.debug(f"- Total images: {cm_result['total_images']}")
            
    elif args.command == 'health':
        result = client.health_check()
        logger.debug(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
