"""
ECG Feature Extraction and Processing

This module handles:
- ECG signal processing (filtering, noise removal)
- R-peak detection
- Heart rate variability (HRV) calculation
- RR interval analysis
- Signal quality assessment
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class ECGProcessor:
    """ECG signal processing and feature extraction."""
    
    def __init__(self, sampling_rate=200):
        """
        Initialize ECG processor.
        
        Parameters:
        -----------
        sampling_rate : int
            ECG sampling rate in Hz (default: 200 Hz)
        """
        self.sampling_rate = sampling_rate
        
    def filter_ecg(self, ecg_signal, lowcut=0.5, highcut=50, order=4):
        """
        Apply bandpass filter to ECG signal.
        
        Parameters:
        -----------
        ecg_signal : array-like
            Raw ECG waveform
        lowcut : float
            Low cutoff frequency (Hz)
        highcut : float
            High cutoff frequency (Hz)
        order : int
            Filter order
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered ECG signal
        """
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure cutoff frequencies are valid
        low = np.clip(low, 0.0001, 0.9999)
        high = np.clip(high, 0.0001, 0.9999)
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        return filtered
    
    def detect_r_peaks(self, ecg_signal, distance=None):
        """
        Detect R-peaks in ECG signal.
        
        Parameters:
        -----------
        ecg_signal : array-like
            Filtered ECG signal
        distance : int or None
            Minimum distance between peaks (in samples).
            If None, calculated from expected HR range (40-150 BPM)
            
        Returns:
        --------
        r_peaks : ndarray
            Indices of detected R-peaks
        """
        if distance is None:
            # For 40-150 BPM range, calculate expected sample distance
            min_hr = 40
            max_hr = 150
            max_distance = int(self.sampling_rate * 60 / min_hr)  # Minimum distance for 40 BPM
            distance = max_distance
        
        # Find peaks in the ECG signal
        r_peaks, _ = find_peaks(ecg_signal, distance=distance, prominence=0.1)
        
        return r_peaks
    
    def calculate_rr_intervals(self, r_peaks):
        """
        Calculate RR intervals from R-peak indices.
        
        Parameters:
        -----------
        r_peaks : array-like
            Indices of R-peaks
            
        Returns:
        --------
        rr_intervals : ndarray
            RR intervals in milliseconds
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / self.sampling_rate) * 1000
        
        return rr_ms
    
    def calculate_hrv_features(self, rr_intervals):
        """
        Calculate Heart Rate Variability (HRV) features.
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in milliseconds
            
        Returns:
        --------
        dict : HRV features
            - ecg_hr_bpm: Heart rate from ECG
            - ecg_rr_interval_ms: Mean RR interval
            - ecg_rmssd_ms: Root Mean Square of Successive Differences
            - ecg_sdnn_ms: Standard Deviation of NN intervals
        """
        if len(rr_intervals) < 2:
            return {
                'ecg_hr_bpm': np.nan,
                'ecg_rr_interval_ms': np.nan,
                'ecg_rmssd_ms': np.nan,
                'ecg_sdnn_ms': np.nan
            }
        
        # Heart rate from RR interval
        mean_rr_ms = np.mean(rr_intervals)
        ecg_hr_bpm = 60000 / mean_rr_ms if mean_rr_ms > 0 else np.nan
        
        # RMSSD: Root Mean Square of Successive Differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # SDNN: Standard Deviation of NN intervals
        sdnn = np.std(rr_intervals)
        
        return {
            'ecg_hr_bpm': ecg_hr_bpm,
            'ecg_rr_interval_ms': mean_rr_ms,
            'ecg_rmssd_ms': rmssd,
            'ecg_sdnn_ms': sdnn
        }
    
    def assess_signal_quality(self, ecg_signal, r_peaks):
        """
        Assess ECG signal quality.
        
        Parameters:
        -----------
        ecg_signal : array-like
            Filtered ECG signal
        r_peaks : array-like
            Indices of detected R-peaks
            
        Returns:
        --------
        quality : float
            Signal quality score (0-100)
        """
        if len(r_peaks) < 2:
            return 0.0
        
        # Calculate signal-to-noise ratio
        signal_power = np.mean(ecg_signal ** 2)
        
        # Estimate noise as variance between peaks
        noise_segments = []
        for i in range(len(r_peaks) - 1):
            start = r_peaks[i] + int(self.sampling_rate * 0.1)  # Skip immediate peak region
            end = r_peaks[i + 1] - int(self.sampling_rate * 0.1)
            
            if start < end:
                noise_segments.append(ecg_signal[start:end])
        
        if noise_segments:
            noise_power = np.mean([np.var(seg) for seg in noise_segments])
            snr = signal_power / (noise_power + 1e-10)
            quality = min(100, 10 * np.log10(snr + 1))
        else:
            quality = 50.0  # Default quality if calculation fails
        
        return float(np.clip(quality, 0, 100))
    
    def process_ecg_segment(self, ecg_signal):
        """
        Complete ECG processing pipeline.
        
        Parameters:
        -----------
        ecg_signal : array-like
            Raw ECG waveform
            
        Returns:
        --------
        dict : All extracted ECG features
        """
        # Step 1: Filter
        filtered = self.filter_ecg(ecg_signal)
        
        # Step 2: Detect R-peaks
        r_peaks = self.detect_r_peaks(filtered)
        
        # Step 3: Calculate RR intervals
        rr_intervals = self.calculate_rr_intervals(r_peaks)
        
        # Step 4: Calculate HRV features
        hrv_features = self.calculate_hrv_features(rr_intervals)
        
        # Step 5: Assess signal quality
        signal_quality = self.assess_signal_quality(filtered, r_peaks)
        
        hrv_features['ecg_signal_quality'] = signal_quality
        
        return hrv_features
    
    def generate_synthetic_ecg(self, duration_seconds=10, heart_rate=75):
        """
        Generate synthetic ECG signal for testing.
        
        Parameters:
        -----------
        duration_seconds : float
            Duration of signal in seconds
        heart_rate : float
            Heart rate in BPM
            
        Returns:
        --------
        ecg_signal : ndarray
            Synthetic ECG signal
        """
        samples = int(duration_seconds * self.sampling_rate)
        t = np.linspace(0, duration_seconds, samples)
        
        # QRS complex (simplified)
        qrs_frequency = heart_rate / 60
        qrs = signal.square(2 * np.pi * qrs_frequency * t, duty=0.3)
        
        # P wave
        p_wave = 0.2 * np.sin(2 * np.pi * qrs_frequency * t + np.pi)
        
        # T wave
        t_wave = 0.3 * np.sin(2 * np.pi * qrs_frequency * t + 3 * np.pi)
        
        # Combine
        ecg = qrs + p_wave + t_wave + np.random.normal(0, 0.1, samples)
        
        return ecg


# Example usage for real-time or batch processing
def demo_ecg_processing():
    """Demonstrate ECG processing pipeline."""
    
    processor = ECGProcessor(sampling_rate=200)
    
    print("="*80)
    print("ECG PROCESSING DEMO")
    print("="*80)
    
    # Generate synthetic ECG
    print("\n[1] Generating synthetic ECG signal...")
    ecg_signal = processor.generate_synthetic_ecg(duration_seconds=30, heart_rate=75)
    print(f"  Generated {len(ecg_signal)} samples")
    
    # Process ECG
    print("\n[2] Processing ECG signal...")
    features = processor.process_ecg_segment(ecg_signal)
    
    print("\n[3] Extracted Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.2f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    demo_ecg_processing()
