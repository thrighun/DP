import cv2
import numpy as np
import tifffile
import os
from datetime import datetime
from skimage import exposure, morphology
from scipy import ndimage
import matplotlib.pyplot as plt

class AdvancedSEMContactAnalyzer:
    def __init__(self, debug=True, save_debug_images=False):
        self.debug = debug and self._is_display_available()
        self.save_debug_images = save_debug_images
        self.debug_image_count = 0
        self.debug_image_dir = "debug_analysis"
        
        # Detection parameters
        self.min_contour_area = 5          # Minimum area in pixels
        self.max_contour_area = 5000       # Maximum area in pixels
        self.circularity_thresh = 0.3      # Minimum circularity (0-1)
        self.contrast_limit = 1.5         # CLAHE contrast limit
        self.calibration_factor = 0.925    # nm/pixel conversion
        
        # Size filtering (in nm)
        self.min_hole_diameter = 10        
        self.max_hole_diameter = 100
        
        # ROI filtering (fraction of image dimensions)
        self.roi_top = 0.05               # Exclude top 10% of image
        self.roi_bottom = 0.9            # Exclude bottom 10% of image
        self.roi_left = 0.05               # Exclude left 10% of image
        self.roi_right = 0.85              # Exclude right 10% of image
        
        # Advanced processing parameters
        self.adaptive_block_size = 101     # Should be odd number
        self.adaptive_c = 4               # Constant subtracted from mean
        self.morph_open_iter = 2          # Morphological opening iterations
        self.morph_close_iter = 2         # Morphological closing iterations
        self.edge_sigma = 1.0             # Edge detection sigma
        self.ellipse_fit_min_points = 10  # Minimum points for ellipse fitting
        
        # New parameters for empty region rejection
        self.max_hole_intensity = 1000      # Maximum allowed mean intensity in hole
        self.min_contrast = 10            # Minimum contrast with surroundings
        
        self.window_name = "SEM Contact Analysis"
        self.contour_color = (0, 255, 0)  # Green
        self.ellipse_color = (0, 0, 255)  # Red
        self.text_color = (255, 255, 255) # White
        
        if save_debug_images:
            os.makedirs(self.debug_image_dir, exist_ok=True)

    def _is_display_available(self):
        try:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test")
            return True
        except:
            return False

    def _debug_show(self, img, title):
        if not self.debug:
            if self.save_debug_images:
                self._save_debug_image(img, title)
            return True

        try:
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img.copy()
                
            cv2.putText(display_img, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(title, display_img)
            key = cv2.waitKey(0)
            
            if self.save_debug_images:
                self._save_debug_image(img, title)
                
            try:
                cv2.destroyWindow(title)
            except:
                pass
                
            return key != ord('q')
        except Exception as e:
            print(f"Debug display failed: {str(e)}")
            if self.save_debug_images:
                self._save_debug_image(img, title)
            return True

    def _save_debug_image(self, img, title):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.debug_image_dir}/{self.debug_image_count:03d}_{title}_{timestamp}.png"
            cv2.imwrite(filename, img)
            self.debug_image_count += 1
        except Exception as e:
            print(f"Error saving debug image: {e}")

    def _adaptive_contrast_normalization(self, img):
        """Enhanced contrast normalization with CLAHE and percentile stretching"""
        clahe = cv2.createCLAHE(clipLimit=self.contrast_limit, tileGridSize=(16,16))
        normalized = clahe.apply(img)

        p2, p98 = np.percentile(normalized, (2, 98))
        normalized = exposure.rescale_intensity(normalized, in_range=(p2, p98))
        
        return cv2.bilateralFilter(normalized, 9, 75, 75)

    def _remove_artifacts(self, img):
        """Improved artifact removal with morphological operations"""
        # Remove bright artifacts
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.dilate(thresh, kernel, iterations=2)
        
        # Remove dark artifacts
        _, dark_thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.dilate(dark_thresh, kernel, iterations=2)
        
        combined_mask = cv2.bitwise_or(mask, dark_mask)
        return cv2.inpaint(img, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def _enhance_holes(self, img):
        """Specialized hole enhancement using top-hat transform"""
        kernel_size = max(3, int(min(img.shape[:2]) * 0.01) // 2 * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Top-hat transform to enhance dark holes
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        
        # Combine with original image
        enhanced = cv2.addWeighted(img, 0.75, tophat, 0.25, 0)
        return enhanced

    def preprocess_image(self, img):
        """Enhanced preprocessing pipeline"""
        try:
            # Convert to 8-bit if needed
            if img.dtype == np.uint16:
                img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
            
            # Initial cleaning
            clean_img = self._remove_artifacts(img)
            if self.debug:
                self._debug_show(clean_img, "01_Cleaned")
            
            # Hole enhancement
            enhanced = self._enhance_holes(clean_img)
            if self.debug:
                self._debug_show(enhanced, "02_Enhanced")
            
            # Contrast normalization
            normalized = self._adaptive_contrast_normalization(enhanced)
            if self.debug:
                self._debug_show(normalized, "03_Normalized")
            
            # Edge-preserving smoothing
            blur = cv2.bilateralFilter(normalized, 9, 75, 75)
            
            # Adaptive thresholding with dynamic block size
            block_size = min(blur.shape[:2]) // 8 * 2 + 1
            block_size = max(3, min(block_size, self.adaptive_block_size))
            
            thresh = cv2.adaptiveThreshold(
                blur, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                block_size, self.adaptive_c
            )
            if self.debug:
                self._debug_show(thresh, "04_Thresholded")
            
            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iter)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter)
            
            # Remove small objects
            cleaned = morphology.remove_small_objects(cleaned.astype(bool), min_size=self.min_contour_area)
            cleaned = cleaned.astype(np.uint8) * 255
            
            if self.debug:
                self._debug_show(cleaned, "05_FinalBinary")
            
            return cleaned
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None

    def _evaluate_contour(self, cnt, img_height, img_width, img_gray):
        """Enhanced contour evaluation with intensity and contrast checks"""
        try:
            area = cv2.contourArea(cnt)
            if area < 1:
                return {'is_valid': False}
            
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Create a mask for the contour
            mask = np.zeros_like(img_gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Calculate mean intensity within the contour
            mean_intensity = cv2.mean(img_gray, mask=mask)[0]
            
            # Calculate contrast with surrounding area
            dilated_mask = cv2.dilate(mask, np.ones((5,5)), iterations=2)
            surround_mask = cv2.subtract(dilated_mask, mask)
            surround_intensity = cv2.mean(img_gray, mask=surround_mask)[0]
            contrast = abs(mean_intensity - surround_intensity)
            
            # Initial diameter estimate
            equivalent_diameter = np.sqrt(4 * area / np.pi) * self.calibration_factor
            major = minor = equivalent_diameter
            ellipticity = 0.0
            
            # Enhanced ellipse fitting
            try:
                if len(cnt) >= self.ellipse_fit_min_points:
                    ellipse = cv2.fitEllipse(cnt)
                    (_, _), (d1, d2), _ = ellipse
                    major_ellipse = max(d1, d2) * self.calibration_factor
                    minor_ellipse = min(d1, d2) * self.calibration_factor
                    
                    rect = cv2.minAreaRect(cnt)
                    rect_major = max(rect[1]) * self.calibration_factor
                    rect_minor = min(rect[1]) * self.calibration_factor
                    
                    major = 0.7 * major_ellipse + 0.3 * rect_major
                    minor = 0.7 * minor_ellipse + 0.3 * rect_minor
                
                ellipticity = 1 - (minor / major) if major > 0 else 0
                
            except Exception as e:
                if self.debug:
                    print(f"Ellipse fitting error: {str(e)}")
            
            # Get center for ROI check
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = np.mean(cnt.squeeze(), axis=0)
            
            # Combined validation with multiple criteria
            is_valid = (
                self.min_contour_area <= area <= self.max_contour_area and
                circularity >= self.circularity_thresh and
                (cy > (img_height * self.roi_top)) and
                (cy < (img_height * self.roi_bottom)) and
                (cx > (img_width * self.roi_left)) and
                (cx < (img_width * self.roi_right)) and
                self.min_hole_diameter <= minor <= major <= self.max_hole_diameter and
                mean_intensity < self.max_hole_intensity and
                contrast > self.min_contrast
            )
            
            return {
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'ellipticity': ellipticity,
                'major': major,
                'minor': minor,
                'diameter': (major + minor) / 2,
                'is_valid': is_valid,
                'cx': cx,
                'cy': cy,
                'mean_intensity': mean_intensity,
                'contrast': contrast
            }
            
        except Exception as e:
            print(f"Contour evaluation failed: {str(e)}")
            return {'is_valid': False}

    def analyze_contours(self, img, contours):
        """Enhanced contour analysis with detailed reporting and 4-way ROI"""
        results = []
        valid_contours = []
        img_height, img_width = img.shape[:2]
        
        # Ensure we have a grayscale image for intensity analysis
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        print("\nContact Hole Analysis Report:")
        print("ID  Center (x,y)   Diameter   Major    Minor    Intensity  Contrast  Circularity  Status")
        print("----------------------------------------------------------------------------------------")
        
        for i, cnt in enumerate(contours):
            try:
                evaluation = self._evaluate_contour(cnt, img_height, img_width, img_gray)
                
                if not evaluation or not evaluation.get('is_valid', False):
                    reject_reasons = []
                    if evaluation.get('area', 0) < self.min_contour_area: 
                        reject_reasons.append(f"small-area({evaluation.get('area', 0):.1f})")
                    if evaluation.get('area', 0) > self.max_contour_area: 
                        reject_reasons.append(f"large-area({evaluation.get('area', 0):.1f})")
                    if evaluation.get('circularity', 0) < self.circularity_thresh: 
                        reject_reasons.append(f"low-circ({evaluation.get('circularity', 0):.2f})")
                    if evaluation.get('cy', img_height) <= (img_height * self.roi_top): 
                        reject_reasons.append("above-top-ROI")
                    if evaluation.get('cy', img_height) >= (img_height * self.roi_bottom): 
                        reject_reasons.append("below-bottom-ROI")
                    if evaluation.get('cx', img_width) <= (img_width * self.roi_left): 
                        reject_reasons.append("left-of-left-ROI")
                    if evaluation.get('cx', img_width) >= (img_width * self.roi_right): 
                        reject_reasons.append("right-of-right-ROI")
                    if evaluation.get('minor', 0) < self.min_hole_diameter: 
                        reject_reasons.append(f"too-small({evaluation.get('minor', 0):.1f}nm)")
                    if evaluation.get('major', 0) > self.max_hole_diameter: 
                        reject_reasons.append(f"too-large({evaluation.get('major', 0):.1f}nm)")
                    if evaluation.get('mean_intensity', 255) >= self.max_hole_intensity: 
                        reject_reasons.append(f"too-bright({evaluation.get('mean_intensity', 0):.1f})")
                    if evaluation.get('contrast', 0) <= self.min_contrast: 
                        reject_reasons.append(f"low-contrast({evaluation.get('contrast', 0):.1f})")
                    
                    print(f"{i:<3} {' '.join(reject_reasons):<70} REJECTED")
                    continue
                    
                results.append({
                    'id': len(results),
                    'center': (evaluation.get('cx', 0), evaluation.get('cy', 0)),
                    'diameter': evaluation.get('diameter', 0),
                    'major': evaluation.get('major', 0),
                    'minor': evaluation.get('minor', 0),
                    'ellipticity': evaluation.get('ellipticity', 0),
                    'circularity': evaluation.get('circularity', 0),
                    'mean_intensity': evaluation.get('mean_intensity', 0),
                    'contrast': evaluation.get('contrast', 0),
                    'contour': cnt,
                    'is_valid': True
                })
                valid_contours.append(cnt)
                
                print(f"{i:<3} ({evaluation.get('cx', 0):.1f},{evaluation.get('cy', 0):.1f}) {evaluation.get('diameter', 0):>7.2f}  "
                      f"{evaluation.get('major', 0):>7.2f}  {evaluation.get('minor', 0):>7.2f}  "
                      f"{evaluation.get('mean_intensity', 0):>8.1f}  {evaluation.get('contrast', 0):>8.1f}  "
                      f"{evaluation.get('circularity', 0):>11.4f}  ACCEPTED")
            
            except Exception as e:
                print(f"Error processing contour {i}: {str(e)}")
                continue
        
        stats = self._calculate_statistics(results)
        return results, valid_contours, stats

    def _calculate_statistics(self, results):
        """Enhanced statistics calculation with percentiles"""
        if not results:
            return None
            
        try:
            valid_results = [r for r in results if isinstance(r, dict) and r.get('is_valid', False)]
            if not valid_results:
                return None
                
            diameters = [r.get('diameter', 0) for r in valid_results]
            majors = [r.get('major', 0) for r in valid_results]
            minors = [r.get('minor', 0) for r in valid_results]
            ellipticities = [r.get('ellipticity', 0) for r in valid_results]
            circularities = [r.get('circularity', 0) for r in valid_results]
            intensities = [r.get('mean_intensity', 0) for r in valid_results]
            contrasts = [r.get('contrast', 0) for r in valid_results]
            
            return {
                'count': len(valid_results),
                'diameter_avg': np.mean(diameters) if diameters else 0,
                'diameter_std': np.std(diameters) if len(diameters) > 1 else 0,
                'diameter_median': np.median(diameters) if diameters else 0,
                'diameter_p10': np.percentile(diameters, 10) if diameters else 0,
                'diameter_p90': np.percentile(diameters, 90) if diameters else 0,
                'major_avg': np.mean(majors) if majors else 0,
                'major_std': np.std(majors) if len(majors) > 1 else 0,
                'minor_avg': np.mean(minors) if minors else 0,
                'minor_std': np.std(minors) if len(minors) > 1 else 0,
                'ellipticity_avg': np.mean(ellipticities) if ellipticities else 0,
                'ellipticity_std': np.std(ellipticities) if len(ellipticities) > 1 else 0,
                'circularity_avg': np.mean(circularities) if circularities else 0,
                'circularity_std': np.std(circularities) if len(circularities) > 1 else 0,
                'intensity_avg': np.mean(intensities) if intensities else 0,
                'intensity_std': np.std(intensities) if len(intensities) > 1 else 0,
                'contrast_avg': np.mean(contrasts) if contrasts else 0,
                'contrast_std': np.std(contrasts) if len(contrasts) > 1 else 0,
                'uniformity_index': (1 - (np.std(diameters)/np.mean(diameters))) if diameters and np.mean(diameters) > 0 else 0
            }
        except Exception as e:
            print(f"Statistics calculation error: {str(e)}")
            return None

    def print_statistics(self, stats):
        """Enhanced statistics printing with more metrics"""
        if not stats:
            print("\nNo valid statistics could be calculated")
            return
            
        print("\n=== COMPREHENSIVE FEATURE STATISTICS ===")
        print(f"Number of features detected: {stats.get('count', 0)}")
        
        print("\nDiameter (nm):")
        print(f"  Average: {stats.get('diameter_avg', 0):.2f} ± {stats.get('diameter_std', 0):.2f}")
        print(f"  Median: {stats.get('diameter_median', 0):.2f}")
        print(f"  10-90 Percentile: {stats.get('diameter_p10', 0):.2f} - {stats.get('diameter_p90', 0):.2f}")
        
        print("\nMajor Axis (nm):")
        print(f"  Average: {stats.get('major_avg', 0):.2f} ± {stats.get('major_std', 0):.2f}")
        
        print("\nMinor Axis (nm):")
        print(f"  Average: {stats.get('minor_avg', 0):.2f} ± {stats.get('minor_std', 0):.2f}")
        
        print("\nEllipticity (1 - minor/major):")
        print(f"  Average: {stats.get('ellipticity_avg', 0):.4f} ± {stats.get('ellipticity_std', 0):.4f}")
        
        print("\nCircularity (4π*area/perimeter²):")
        print(f"  Average: {stats.get('circularity_avg', 0):.4f} ± {stats.get('circularity_std', 0):.4f}")
        
        print("\nIntensity (0-255):")
        print(f"  Average: {stats.get('intensity_avg', 0):.1f} ± {stats.get('intensity_std', 0):.1f}")
        
        print("\nContrast (0-255):")
        print(f"  Average: {stats.get('contrast_avg', 0):.1f} ± {stats.get('contrast_std', 0):.1f}")
        
        print("\nUniformity Index (1 - CV):")
        print(f"  {stats.get('uniformity_index', 0):.4f}")

    def save_results_to_csv(self, results, stats, image_path):
        """Enhanced CSV saving with more fields"""
        try:
            csv_path = os.path.join(self.debug_image_dir, "analysis_results.csv")
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, 'a') as f:
                if not file_exists:
                    f.write("timestamp,image_path,id,center_x,center_y,diameter,major,minor,ellipticity,circularity,intensity,contrast,area\n")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for r in results:
                    if not isinstance(r, dict) or not r.get('is_valid', False):
                        continue
                    f.write(f"{timestamp},{image_path},{r.get('id', '')},{r.get('center', (0,0))[0]:.2f},{r.get('center', (0,0))[1]:.2f},")
                    f.write(f"{r.get('diameter', 0):.2f},{r.get('major', 0):.2f},{r.get('minor', 0):.2f},")
                    f.write(f"{r.get('ellipticity', 0):.4f},{r.get('circularity', 0):.4f},")
                    f.write(f"{r.get('mean_intensity', 0):.1f},{r.get('contrast', 0):.1f},{r.get('area', 0):.2f}\n")
                
                if stats:
                    f.write("\nSTATISTICS_SUMMARY\n")
                    f.write(f"total_features,{stats.get('count', 0)}\n")
                    f.write(f"diameter_avg,{stats.get('diameter_avg', 0):.4f}\n")
                    f.write(f"diameter_std,{stats.get('diameter_std', 0):.4f}\n")
                    f.write(f"diameter_median,{stats.get('diameter_median', 0):.4f}\n")
                    f.write(f"diameter_p10,{stats.get('diameter_p10', 0):.4f}\n")
                    f.write(f"diameter_p90,{stats.get('diameter_p90', 0):.4f}\n")
                    f.write(f"major_axis_avg,{stats.get('major_avg', 0):.4f}\n")
                    f.write(f"major_axis_std,{stats.get('major_std', 0):.4f}\n")
                    f.write(f"minor_axis_avg,{stats.get('minor_avg', 0):.4f}\n")
                    f.write(f"minor_axis_std,{stats.get('minor_std', 0):.4f}\n")
                    f.write(f"ellipticity_avg,{stats.get('ellipticity_avg', 0):.6f}\n")
                    f.write(f"ellipticity_std,{stats.get('ellipticity_std', 0):.6f}\n")
                    f.write(f"circularity_avg,{stats.get('circularity_avg', 0):.6f}\n")
                    f.write(f"circularity_std,{stats.get('circularity_std', 0):.6f}\n")
                    f.write(f"intensity_avg,{stats.get('intensity_avg', 0):.1f}\n")
                    f.write(f"intensity_std,{stats.get('intensity_std', 0):.1f}\n")
                    f.write(f"contrast_avg,{stats.get('contrast_avg', 0):.1f}\n")
                    f.write(f"contrast_std,{stats.get('contrast_std', 0):.1f}\n")
                    f.write(f"uniformity_index,{stats.get('uniformity_index', 0):.6f}\n")
            
            print(f"\nResults saved to: {csv_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def visualize_results(self, img, results, stats):
        """Enhanced visualization with better annotations and ROI visualization"""
        if len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = img.copy()
        
        # Draw ROI boundaries
        height, width = img.shape[:2]
        top_bound = int(height * self.roi_top)
        bottom_bound = int(height * self.roi_bottom)
        left_bound = int(width * self.roi_left)
        right_bound = int(width * self.roi_right)
        
        # Draw ROI rectangle
        cv2.rectangle(display_img, 
                      (left_bound, top_bound), 
                      (right_bound, bottom_bound), 
                      (255, 255, 0), 1)  # Cyan border
        
        # Draw all valid contours and ellipses
        for r in results:
            if not isinstance(r, dict) or not r.get('is_valid', False):
                continue
                
            cnt = r.get('contour')
            center = tuple(map(int, r.get('center', (0,0))))
            diameter = r.get('diameter', 0)
            
            # Draw contour
            cv2.drawContours(display_img, [cnt], -1, self.contour_color, 1)
            
            # Draw fitted ellipse
            if len(cnt) >= self.ellipse_fit_min_points:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(display_img, ellipse, self.ellipse_color, 1)
                except:
                    pass
            
            # Draw center and ID
            cv2.circle(display_img, center, 2, (255,0,0), -1)
            cv2.putText(display_img, f"{r.get('id', '')}", 
                       (center[0]+int(diameter/2)+5, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
        
        # Add statistics overlay
        if stats:
            text_y = 30
            cv2.putText(display_img, f"Contacts: {stats.get('count', 0)}", (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            text_y += 30
            cv2.putText(display_img, f"Avg Diam: {stats.get('diameter_avg', 0):.1f} ± {stats.get('diameter_std', 0):.1f} nm", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            text_y += 30
            cv2.putText(display_img, f"Circularity: {stats.get('circularity_avg', 0):.2f}", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            text_y += 30
            cv2.putText(display_img, f"Intensity: {stats.get('intensity_avg', 0):.1f} ± {stats.get('intensity_std', 0):.1f}", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return display_img

    def analyze(self, image_path):
        """Enhanced analysis pipeline with better visualization"""
        print(f"\nAnalyzing {image_path}")
        
        try:
            img = tifffile.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
                
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            processed = self.preprocess_image(img)
            if processed is None:
                raise ValueError("Image preprocessing failed")
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            print(f"\nFound {len(contours)} potential contacts")
            
            # Analyze contours
            results, valid_contours, stats = self.analyze_contours(img, contours)
            
            if results:
                self.print_statistics(stats)
                
                # Visualize results
                display_img = self.visualize_results(img, results, stats)
                
                # Show and save results
                if self.debug:
                    cv2.imshow(self.window_name, display_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                if self.save_debug_images:
                    self.save_results_to_csv(results, stats, image_path)
                    cv2.imwrite(os.path.join(self.debug_image_dir, "final_result.png"), display_img)
                
                return results, stats
            else:
                print("\nNo valid contacts found - Suggested adjustments:")
                print(f"- min_contour_area (current: {self.min_contour_area})")
                print(f"- circularity_thresh (current: {self.circularity_thresh})")
                print(f"- min_hole_diameter (current: {self.min_hole_diameter}nm)")
                print(f"- max_hole_diameter (current: {self.max_hole_diameter}nm)")
                print(f"- max_hole_intensity (current: {self.max_hole_intensity})")
                print(f"- min_contrast (current: {self.min_contrast})")
                print("- Check if preprocessing is removing actual contacts")
                return None, None
                
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
            return None, None

    def batch_analyze(self, image_folder, output_csv="batch_results.csv"):
        """Batch process all images in a folder"""
        try:
            image_files = [f for f in os.listdir(image_folder) 
                         if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"No images found in {image_folder}")
                return
            
            all_results = []
            
            for img_file in image_files:
                img_path = os.path.join(image_folder, img_file)
                results, stats = self.analyze(img_path)
                
                if results and stats:
                    stats['image'] = img_file
                    all_results.append(stats)
            
            if all_results:
                self._save_batch_results(all_results, output_csv)
                print(f"\nBatch analysis complete. Results saved to {output_csv}")
            
        except Exception as e:
            print(f"Batch analysis failed: {str(e)}")

    def _save_batch_results(self, results, output_csv):
        """Save batch analysis results to CSV"""
        try:
            with open(output_csv, 'w') as f:
                # Write header
                f.write("image,count,diameter_avg,diameter_std,diameter_median,")
                f.write("major_avg,major_std,minor_avg,minor_std,")
                f.write("ellipticity_avg,ellipticity_std,circularity_avg,circularity_std,")
                f.write("intensity_avg,intensity_std,contrast_avg,contrast_std,uniformity_index\n")
                
                # Write data
                for r in results:
                    f.write(f"{r.get('image', '')},{r.get('count', 0)},")
                    f.write(f"{r.get('diameter_avg', 0):.2f},{r.get('diameter_std', 0):.2f},{r.get('diameter_median', 0):.2f},")
                    f.write(f"{r.get('major_avg', 0):.2f},{r.get('major_std', 0):.2f},")
                    f.write(f"{r.get('minor_avg', 0):.2f},{r.get('minor_std', 0):.2f},")
                    f.write(f"{r.get('ellipticity_avg', 0):.4f},{r.get('ellipticity_std', 0):.4f},")
                    f.write(f"{r.get('circularity_avg', 0):.4f},{r.get('circularity_std', 0):.4f},")
                    f.write(f"{r.get('intensity_avg', 0):.1f},{r.get('intensity_std', 0):.1f},")
                    f.write(f"{r.get('contrast_avg', 0):.1f},{r.get('contrast_std', 0):.1f},")
                    f.write(f"{r.get('uniformity_index', 0):.4f}\n")
        except Exception as e:
            print(f"Error saving batch results: {str(e)}")


if __name__ == "__main__":
    analyzer = AdvancedSEMContactAnalyzer(
        debug=True,
        save_debug_images=True
    )
    
    image_path = "D:\DP\Images\contacts5.tif"  # Use your image filename
    results, stats = analyzer.analyze(image_path)
    
    # For batch processing:
    # analyzer.batch_analyze("path/to/images/folder")