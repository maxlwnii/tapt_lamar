"""
Preprocess CLIP peak sequences for model pre-training.
Handles variable-length sequences with bucketing strategy:
- Short (<= 1024): pad to 1024
- Medium (1024-10000): sliding windows
- Very long (>10000): weighted sampling around peaks
"""

import numpy as np
from Bio import SeqIO
from collections import defaultdict
import argparse
import json


def read_fasta(fasta_file, filter_random=True):
    """Read FASTA file and return sequences sorted by length.

    If headers include genomic coordinates like "chr1:14262-25035" this parses them
    and stores chrom, seq_start and seq_end so BED peaks can be matched.
    
    Args:
        filter_random: If True, exclude sequences from random/unplaced chromosomes
    """
    import re

    coord_re = re.compile(r'(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)')
    sequences = []
    filtered_count = 0
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.id
        m = coord_re.search(header)
        if m:
            chrom = m.group('chrom')
            seq_start = int(m.group('start'))
            seq_end = int(m.group('end'))
        else:
            chrom = None
            seq_start = 0
            seq_end = len(record.seq) - 1

        # Filter random/unplaced chromosomes
        if filter_random and chrom is not None and not is_standard_chromosome(chrom):
            filtered_count += 1
            continue

        sequences.append({
            'id': header,
            'seq': str(record.seq).upper(),
            'length': len(record.seq),
            'chrom': chrom,
            'seq_start': seq_start,
            'seq_end': seq_end
        })

    # Sort by length
    sequences.sort(key=lambda x: x['length'])
    print(f"Loaded {len(sequences)} sequences")
    if filter_random and filtered_count > 0:
        print(f"Filtered out {filtered_count} sequences from random/unplaced chromosomes")
    if sequences:
        print(f"Length range: {sequences[0]['length']} - {sequences[-1]['length']}")
    return sequences


def is_standard_chromosome(chrom):
    """Check if chromosome is a standard chromosome (exclude random/unplaced)."""
    chrom_lower = chrom.lower()
    
    # Exclude random, unplaced, alt, fix, or decoy sequences
    if any([
        '_random' in chrom_lower,
        'un_' in chrom_lower,
        'chrun' in chrom_lower,
        '_alt' in chrom_lower,
        '_fix' in chrom_lower,
        '_decoy' in chrom_lower,
    ]):
        return False
    
    # Include standard chromosomes (chr1-22, chrX, chrY, chrM)
    # Remove 'chr' prefix for checking
    chrom_num = chrom.replace('chr', '').replace('Chr', '')
    
    if chrom_num in ['X', 'Y', 'M', 'MT']:
        return True
    
    try:
        num = int(chrom_num)
        return 1 <= num <= 22
    except ValueError:
        return False


def read_bed_file(bed_file, filter_random=True):
    """Read BED file and return peak positions per chromosome.
    
    Args:
        bed_file: Path to BED file
        filter_random: If True, exclude random/unplaced chromosomes
    """
    peaks = defaultdict(list)
    filtered_count = 0
    
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split('\t')
            chrom = parts[0]
            
            # Filter random/unplaced chromosomes
            if filter_random and not is_standard_chromosome(chrom):
                filtered_count += 1
                continue
            
            start = int(parts[1])
            end = int(parts[2])
            bindingsites = int(parts[4])
            print(f"Peak: {chrom}:{start}-{end} (number of Peaks: {bindingsites})")
            peaks[chrom].append({
                'start': start,
                'end': end,
                'bindingsites': bindingsites     
            })
    
    print(f"Loaded peaks for {len(peaks)} chromosomes")
    if filter_random and filtered_count > 0:
        print(f"Filtered out {filtered_count} peaks from random/unplaced chromosomes")
    
    for chrom in sorted(peaks.keys()):
        print(f"  {chrom}: {len(peaks[chrom])} peaks")
    
    return peaks


def sliding_window(seq, seq_id, peaks, max_len=1024, stride=256):
    """Generate sliding windows from sequence.
    
    Args:
        seq: DNA sequence string
        seq_id: Sequence identifier
        peaks: List of peaks in sequence-local coordinates
        max_len: Window length
        stride: Sliding window stride
    """
    windows = []
    seq_len = len(seq)
    
    for i in range(0, seq_len - max_len + 1, stride):
        window = seq[i:i+max_len]
        
        # Count peaks in this window
        peaks_in_window = sum(1 for p in peaks 
                             if i <= p['bindingsites'] < i + max_len)
        
        windows.append({
            'sequence': window,
            'seq_id': seq_id,
            'window_start': i,
            'window_end': i + max_len,
            'method': 'sliding_window',
            'peak_coverage': peaks_in_window
        })
    
    # Handle the last window if it doesn't fit perfectly
    if seq_len % stride != 0 and seq_len > max_len:
        last_start = seq_len - max_len
        if last_start > windows[-1]['window_start']:  # Avoid duplicate
            # Count peaks in last window
            peaks_in_last = sum(1 for p in peaks 
                               if last_start <= p['bindingsites'] < seq_len)
            
            windows.append({
                'sequence': seq[last_start:],
                'seq_id': seq_id,
                'window_start': last_start,
                'window_end': seq_len,
                'method': 'sliding_window',
                'peak_coverage': peaks_in_last
            })
    
    return windows


def weighted_sampling_around_peaks(seq, seq_id, peaks, max_len=1024, n_samples=None):
    """Sample windows around peak regions with weighted probability."""
    windows = []
    seq_len = len(seq)
    
    if not peaks:
        print(f"Warning: No peaks found for {seq_id}, using random sampling")
        # Fallback to random sampling
        n_samples = n_samples or min(10, max(3, seq_len // 5000))
        for _ in range(n_samples):
            start = np.random.randint(0, max(1, seq_len - max_len + 1))
            windows.append({
                'sequence': seq[start:start+max_len],
                'seq_id': seq_id,
                'window_start': start,
                'window_end': min(start + max_len, seq_len),
                'method': 'random_sampling',
                'peak_coverage': 0
            })
        return windows
    
    # Filter peaks that are within sequence bounds
    valid_peaks = [p for p in peaks if 0 <= p['center'] < seq_len]
    if not valid_peaks:
        print(f"Warning: No valid peaks within bounds for {seq_id}, using random sampling")
        n_samples = n_samples or min(10, max(3, seq_len // 5000))
        for _ in range(n_samples):
            start = np.random.randint(0, max(1, seq_len - max_len + 1))
            windows.append({
                'sequence': seq[start:start+max_len],
                'seq_id': seq_id,
                'window_start': start,
                'window_end': min(start + max_len, seq_len),
                'method': 'random_sampling',
                'peak_coverage': 0
            })
        return windows
    
    # Calculate number of samples based on number of peaks
    if n_samples is None:
        n_samples = min(10, max(len(valid_peaks), int(2 * np.sqrt(len(valid_peaks)))))
    
    # Ensure at least one window per peak (up to n_samples)
    peak_centers = [p['bindingsites'] for p in valid_peaks]
    sampled_peaks = np.random.choice(len(peak_centers), 
                                     size=min(n_samples, len(peak_centers)), 
                                     replace=False)
    
    for peak_idx in sampled_peaks:
        peak_center = peak_centers[peak_idx]
        
        # Add jitter around peak center (±max_len/2)
        jitter = np.random.randint(-max_len//2, max_len//2)
        window_center = np.clip(peak_center + jitter, max_len//2, seq_len - max_len//2)
        
        start = window_center - max_len//2
        start = max(0, min(start, seq_len - max_len))
        end = min(start + max_len, seq_len)
        
        # Count peaks in this window
        peaks_in_window = sum(1 for p in valid_peaks 
                             if start <= p['bindingsites'] <= end)
        
        windows.append({
            'sequence': seq[start:end],
            'seq_id': seq_id,
            'window_start': start,
            'window_end': end,
            'method': 'peak_sampling',
            'peak_coverage': peaks_in_window,
            'target_peak': peak_idx
        })
    
    # Add additional random samples if needed
    remaining_samples = n_samples - len(windows)
    if remaining_samples > 0:
        for _ in range(remaining_samples):
            # Sample with probability proportional to distance from peaks
            weights = np.ones(seq_len)
            for peak in valid_peaks:
                center = peak['bindingsites']
                # Increase weight around peaks (Gaussian-like)
                distances = np.abs(np.arange(seq_len) - center)
                weights += np.exp(-distances / (max_len / 2))
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Sample a center position
            window_center = np.random.choice(seq_len, p=weights)
            start = max(0, min(window_center - max_len//2, seq_len - max_len))
            end = min(start + max_len, seq_len)
            
            peaks_in_window = sum(1 for p in valid_peaks 
                                 if start <= p['bindingsites'] <= end)
            
            windows.append({
                'sequence': seq[start:end],
                'seq_id': seq_id,
                'window_start': start,
                'window_end': end,
                'method': 'weighted_random',
                'peak_coverage': peaks_in_window
            })
    
    return windows


def preprocess_sequences(sequences, peaks_dict, max_len=1024, stride=256):
    """Main preprocessing function.

    Matches BED peaks (chromosomal coordinates) to FASTA sequences by chromosome
    and coordinate range. Converts matched peaks to sequence-local coordinates
    before calling sampling routines.
    """
    processed_data = []
    stats = {
        'short_padded': 0,
        'medium_windowed': 0,
        'long_sampled': 0,
        'total_windows': 0
    }

    for seq_record in sequences:
        seq_id = seq_record['id']
        seq = seq_record['seq']
        seq_len = seq_record['length']

        # Determine chromosome and genomic interval for this sequence (if available)
        chrom = seq_record.get('chrom', None)
        seq_start = seq_record.get('seq_start', 0)
        seq_end = seq_record.get('seq_end', seq_len - 1)

        # Gather peaks overlapping this sequence and convert to local coordinates
        seq_peaks_rel = []
        if chrom is not None and chrom in peaks_dict:
            raw_peaks = peaks_dict[chrom]
            for p in raw_peaks:
                if p['bindingsites'] >= seq_start and p['bindingsites'] <= seq_end:
                    # convert to 0-based local coordinate within seq
                    local_start = max(0, p['start'] - seq_start)
                    local_end = min(seq_len, p['end'] - seq_start)
                    local_center = p['center'] - seq_start
                    # keep only peaks with a valid local center
                    if 0 <= local_center < seq_len:
                        seq_peaks_rel.append({
                            'start': local_start,
                            'end': local_end,
                            'center': local_center
                        })

        if seq_len <= max_len:
            # SHORT: Keep original length, tokenizer will pad during training
            # Count all peaks in the sequence
            peak_count = len(seq_peaks_rel)
            
            processed_data.append({
                'sequence': seq,  # no manual padding
                'seq_id': seq_id,
                'original_length': seq_len,
                'window_start': 0,
                'window_end': seq_len,
                'method': 'short_sequence',
                'peak_coverage': peak_count
            })
            stats['short_padded'] += 1
            stats['total_windows'] += 1

        elif seq_len <= 10000:
            # MEDIUM: Sliding windows (pass peaks for counting)
            windows = sliding_window(seq, seq_id, seq_peaks_rel, max_len, stride)
            processed_data.extend(windows)
            stats['medium_windowed'] += 1
            stats['total_windows'] += len(windows)

        else:
            # VERY LONG: Weighted sampling around peaks (use sequence-local peaks)
            windows = weighted_sampling_around_peaks(seq, seq_id, seq_peaks_rel, max_len)
            processed_data.extend(windows)
            stats['long_sampled'] += 1
            stats['total_windows'] += len(windows)

    return processed_data, stats


def save_processed_data(processed_data, output_prefix):
    """Save processed sequences and metadata."""
    
    # Save sequences as text file (one per line) for easy loading
    with open(f'{output_prefix}_sequences.txt', 'w') as f:
        for item in processed_data:
            f.write(item['sequence'] + '\n')
    
    # Save metadata as JSON (convert numpy types to Python types)
    metadata = []
    for item in processed_data:
        meta_dict = {}
        for k, v in item.items():
            if k != 'sequence':
                # Convert numpy types to Python native types
                if hasattr(v, 'item'):  # numpy scalar
                    meta_dict[k] = v.item()
                elif isinstance(v, (np.integer, np.int64, np.int32)):
                    meta_dict[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    meta_dict[k] = float(v)
                else:
                    meta_dict[k] = v
        metadata.append(meta_dict)
    
    with open(f'{output_prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved {len(processed_data)} processed sequences to:")
    print(f"  - {output_prefix}_sequences.txt")
    print(f"  - {output_prefix}_metadata.json")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CLIP peak sequences')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--bed', required=True, help='Input BED file with peak regions')
    parser.add_argument('--output', required=True, help='Output prefix for processed files')
    parser.add_argument('--max-len', type=int, default=1024, help='Maximum sequence length (default: 1024)')
    parser.add_argument('--stride', type=int, default=256, help='Stride for sliding windows (default: 256)')
    parser.add_argument('--no-filter', action='store_true', help='Do not filter random/unplaced chromosomes')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CLIP Peak Sequence Preprocessing")
    print("=" * 80)
    
    filter_mode = not args.no_filter
    
    # Read input files
    print("\n[1/4] Reading FASTA file...")
    sequences = read_fasta(args.fasta, filter_random=filter_mode)
    
    print("\n[2/4] Reading BED file...")
    peaks_dict = read_bed_file(args.bed, filter_random=filter_mode)
    
    # Preprocess
    print(f"\n[3/4] Processing sequences (max_len={args.max_len}, stride={args.stride})...")
    processed_data, stats = preprocess_sequences(
        sequences, peaks_dict, 
        max_len=args.max_len, 
        stride=args.stride
    )
    
    # Save results
    print("\n[4/4] Saving processed data...")
    save_processed_data(processed_data, args.output)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("PREPROCESSING STATISTICS")
    print("=" * 80)
    print(f"Short sequences (padded):          {stats['short_padded']:>8}")
    print(f"Medium sequences (windowed):       {stats['medium_windowed']:>8}")
    print(f"Long sequences (peak-sampled):     {stats['long_sampled']:>8}")
    print(f"{'─' * 50}")
    print(f"Total input sequences:             {len(sequences):>8}")
    print(f"Total output windows:              {stats['total_windows']:>8}")
    print(f"Expansion factor:                  {stats['total_windows']/len(sequences):>8.2f}x")
    print("=" * 80)


if __name__ == '__main__':
    main()