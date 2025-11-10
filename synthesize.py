import rawpy
import numpy as np
import tifffile
import subprocess
import os
import imageio
    

def save_modified_raw_with_exif(input_arw, output_tiff):
    """
    Save modified Bayer with metadata preserved (excluding structural tags)
    """
    
    # 1. Read and modify Bayer
    print("Reading and modifying RAW...")
    with rawpy.imread(input_arw) as raw:
        original_bayer = raw.raw_image.copy()
        black_level = raw.black_level_per_channel[0]
        white_level = raw.white_level
        cfa_pattern = raw.raw_pattern
        # Save an image for testing
        rgb = raw.postprocess()
        imageio.imsave('original_test.png', rgb)
        print("✓ Saved: original_test.png")
        print(f"✓ Read original Bayer: {original_bayer.shape}")
        
        print(f"Original range: [{original_bayer.min()}, {original_bayer.max()}]")
        
        # Modify (currently just copying - you can add your processing)
        modified_bayer = original_bayer.copy()
        modified_bayer = modified_bayer.astype(np.float32)
        modified_bayer *= 1.2  # Example modification: increase brightness
        modified_bayer = np.clip(modified_bayer, 0, white_level).astype(np.uint16)
    
    # 2. Save modified Bayer as TIFF first
    print("Saving modified data...")
    cfa_pattern_flat = cfa_pattern.flatten().astype(np.uint8)
    cfa_repeat = np.array([2, 2], dtype=np.uint16)
    
    tifffile.imwrite(
        output_tiff,
        modified_bayer,
        photometric='CFA',
        compression='none',
        planarconfig='contig',
        extratags=[
            (33422, 'B', 4, cfa_pattern_flat, True),
            (33421, 'H', 2, cfa_repeat, True),
        ]
    )
    
    # 3. Copy metadata EXCLUDING structural tags
    print("Copying metadata (excluding structural tags)...")
    result = subprocess.run([
        'exiftool',
        '-TagsFromFile', input_arw,
        '-all:all',
        # EXCLUDE structural tags that break the file
        '--ImageWidth',
        '--ImageHeight', 
        '--ImageLength',
        '--StripOffsets',
        '--StripByteCounts',
        '--TileOffsets',
        '--TileByteCounts',
        '--Compression',
        '--PhotometricInterpretation',
        '--Orientation',
        '--SamplesPerPixel',
        '--PlanarConfiguration',
        '--BitsPerSample',
        '--SubIFD',
        '-overwrite_original',
        output_tiff
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: {result.stderr}")
    else:
        print(f"✓ Metadata copied")
    
    # 4. Verify
    print("\nVerifying...")
    try:
        with rawpy.imread(output_tiff) as raw:
            verify = raw.raw_image
            print(f"✓ Can read Bayer: {verify.shape}")
            print(f"  Range: [{verify.min()}, {verify.max()}]")
            
            # Test postprocess
            rgb = raw.postprocess()
            print(f"✓ Can postprocess: {rgb.shape}")
            print(f"  RGB range: [{rgb.min()}, {rgb.max()}]")
            
            # Save RGB
            imageio.imsave('test_output.png', rgb)
            print("✓ Saved: test_output.png")
            
            return verify, rgb
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

# Usage
bayer, rgb = save_modified_raw_with_exif(
    'images/Sony/short/20211_00_0.033s.ARW',
    'modified_raw.tiff'
)