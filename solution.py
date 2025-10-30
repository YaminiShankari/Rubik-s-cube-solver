# ==========================================================
# 🧩 Rubik's Cube Solver with AUTO-CALIBRATION
# Automatically finds the correct face orientations!
# ==========================================================

import numpy as np
import kociemba
from itertools import product

# ----------------------------------------------------------
# 🎨 Color mapping for Kociemba format
# ----------------------------------------------------------
COLOR_TO_FACE = {
    "white": "U",   # Up
    "red": "R",     # Right
    "green": "F",   # Front
    "yellow": "D",  # Down
    "orange": "L",  # Left
    "blue": "B"     # Back
}

# ----------------------------------------------------------
# 🔄 Rotation Functions
# ----------------------------------------------------------
def flip_horizontal(face):
    """Flip face horizontally (lateral inversion fix)"""
    return np.fliplr(face)

def rotate_0(face):
    """No rotation"""
    return face

def rotate_90_cw(face):
    """Rotate 90° clockwise"""
    return np.rot90(face, k=-1)

def rotate_180(face):
    """Rotate 180°"""
    return np.rot90(face, k=2)

def rotate_270_cw(face):
    """Rotate 270° clockwise (= 90° counter-clockwise)"""
    return np.rot90(face, k=1)

# ----------------------------------------------------------
# 🔄 Apply transformation based on rotation config
# ----------------------------------------------------------
def apply_transformations(captured_faces, rotation_config, lateral_invert=True):
    """
    Apply transformations to all faces based on rotation configuration.
    
    rotation_config: dict with keys 'white', 'red', 'green', 'yellow', 'orange', 'blue'
                     and values 0, 90, 180, or 270 (degrees)
    lateral_invert: if True, flip all faces horizontally first
    """
    
    rotation_funcs = {
        0: rotate_0,
        90: rotate_90_cw,
        180: rotate_180,
        270: rotate_270_cw
    }
    
    transformed = {}
    
    for color in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
        face = np.array(captured_faces[color])
        
        # Step 1: Lateral inversion if needed
        if lateral_invert:
            face = flip_horizontal(face)
        
        # Step 2: Apply rotation
        rotation_degrees = rotation_config[color]
        face = rotation_funcs[rotation_degrees](face)
        
        transformed[color] = face
    
    return transformed

# ----------------------------------------------------------
# 🔄 Convert to Kociemba string
# ----------------------------------------------------------
def faces_to_kociemba_string(transformed_faces):
    """Convert transformed faces to Kociemba string format."""
    
    face_order = ['white', 'red', 'green', 'yellow', 'orange', 'blue']
    kociemba_string = ""
    
    for face_color in face_order:
        face_array = transformed_faces[face_color]
        for row in face_array:
            for color in row:
                color_lower = color.lower()
                if color_lower not in COLOR_TO_FACE:
                    return None  # Invalid color
                kociemba_string += COLOR_TO_FACE[color_lower]
    
    return kociemba_string

# ----------------------------------------------------------
# 🔍 Validate Kociemba string
# ----------------------------------------------------------
def is_valid_kociemba_string(kociemba_string):
    """Check if the Kociemba string is valid by trying to solve it."""
    
    if len(kociemba_string) != 54:
        return False
    
    # Check centers are correct
    centers = {
        'U': kociemba_string[4],
        'R': kociemba_string[13],
        'F': kociemba_string[22],
        'D': kociemba_string[31],
        'L': kociemba_string[40],
        'B': kociemba_string[49]
    }
    
    if not all([
        centers['U'] == 'U',
        centers['R'] == 'R',
        centers['F'] == 'F',
        centers['D'] == 'D',
        centers['L'] == 'L',
        centers['B'] == 'B'
    ]):
        return False
    
    # Try to solve it
    try:
        solution = kociemba.solve(kociemba_string)
        return True
    except:
        return False

# ----------------------------------------------------------
# 🤖 AUTO-CALIBRATION: Find correct rotations
# ----------------------------------------------------------
def auto_calibrate(captured_faces):
    """
    Automatically find the correct rotation configuration.
    Tests all possible combinations with lateral inversion.
    """
    
    print("\n" + "="*70)
    print("🤖 AUTO-CALIBRATION MODE")
    print("="*70)
    print("\n🔍 Searching for correct face orientations...")
    print("   This may take a minute - testing rotation combinations...")
    print()
    
    # Strategy: Most faces are likely to need 0° or 180° rotation
    # So we test those first (common cases)
    common_rotations = [0, 180]
    all_rotations = [0, 90, 180, 270]
    
    # Phase 1: Try common rotations first (faster)
    print("📊 Phase 1: Testing common rotations (0°, 180°)...")
    total_tested = 0
    
    for white_rot in common_rotations:
        for red_rot in all_rotations:  # Red often needs special rotation
            for green_rot in common_rotations:
                for yellow_rot in common_rotations:
                    for orange_rot in all_rotations:  # Orange often needs special rotation
                        for blue_rot in common_rotations:
                            
                            total_tested += 1
                            if total_tested % 100 == 0:
                                print(f"   Tested {total_tested} combinations...")
                            
                            rotation_config = {
                                'white': white_rot,
                                'red': red_rot,
                                'green': green_rot,
                                'yellow': yellow_rot,
                                'orange': orange_rot,
                                'blue': blue_rot
                            }
                            
                            # Apply transformations with lateral inversion
                            transformed = apply_transformations(captured_faces, rotation_config, lateral_invert=True)
                            kociemba_string = faces_to_kociemba_string(transformed)
                            
                            if kociemba_string and is_valid_kociemba_string(kociemba_string):
                                print(f"\n✅ SUCCESS! Found valid configuration after {total_tested} attempts!")
                                print("\n🎯 Correct Rotation Configuration:")
                                print("   (All faces laterally inverted first, then rotated)")
                                print("-"*70)
                                for color in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
                                    print(f"   {color.upper():8s}: {rotation_config[color]:3d}° clockwise")
                                print("-"*70)
                                return rotation_config, transformed, kociemba_string
    
    # Phase 2: If not found, try ALL combinations
    print(f"\n📊 Phase 2: Testing all possible combinations...")
    print("   (This is more thorough but slower)")
    
    for rotations in product(all_rotations, repeat=6):
        total_tested += 1
        if total_tested % 500 == 0:
            print(f"   Tested {total_tested} combinations...")
        
        rotation_config = {
            'white': rotations[0],
            'red': rotations[1],
            'green': rotations[2],
            'yellow': rotations[3],
            'orange': rotations[4],
            'blue': rotations[5]
        }
        
        # Apply transformations with lateral inversion
        transformed = apply_transformations(captured_faces, rotation_config, lateral_invert=True)
        kociemba_string = faces_to_kociemba_string(transformed)
        
        if kociemba_string and is_valid_kociemba_string(kociemba_string):
            print(f"\n✅ SUCCESS! Found valid configuration after {total_tested} attempts!")
            print("\n🎯 Correct Rotation Configuration:")
            print("   (All faces laterally inverted first, then rotated)")
            print("-"*70)
            for color in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
                print(f"   {color.upper():8s}: {rotation_config[color]:3d}° clockwise")
            print("-"*70)
            return rotation_config, transformed, kociemba_string
    
    print(f"\n❌ Could not find valid configuration after {total_tested} attempts.")
    print("   This likely means there are color detection errors in the captured data.")
    return None, None, None

# ----------------------------------------------------------
# 🔍 Validate cube configuration
# ----------------------------------------------------------
def validate_cube(captured_faces):
    """Validates that the cube has exactly 9 stickers of each color."""
    from collections import Counter
    
    all_colors = []
    for face_array in captured_faces.values():
        for row in face_array:
            for color in row:
                all_colors.append(color.lower())
    
    color_counts = Counter(all_colors)
    
    print("\n📊 Color Count Analysis:")
    for color in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
        count = color_counts.get(color, 0)
        status = "✅" if count == 9 else "❌"
        print(f"  {status} {color.capitalize():8s}: {count:2d}/9")
    
    if all(color_counts.get(color, 0) == 9 for color in ['white', 'red', 'green', 'yellow', 'orange', 'blue']):
        print("\n✅ Cube configuration is valid!")
        return True
    else:
        print("\n❌ Invalid cube configuration! Each color should appear exactly 9 times.")
        return False

# ----------------------------------------------------------
# 📖 Format solution for user-friendly output
# ----------------------------------------------------------
def format_solution(solution_string):
    """Formats the Kociemba solution into readable instructions."""
    
    move_descriptions = {
        "U": "Turn the TOP side 90° clockwise",
        "U'": "Turn the TOP side 90° counter-clockwise",
        "U2": "Turn the TOP side 180°",
        "D": "Turn the BOTTOM side 90° clockwise",
        "D'": "Turn the BOTTOM side 90° counter-clockwise",
        "D2": "Turn the BOTTOM side 180°",
        "L": "Turn the LEFT side 90° clockwise",
        "L'": "Turn the LEFT side 90° counter-clockwise",
        "L2": "Turn the LEFT side 180°",
        "R": "Turn the RIGHT side 90° clockwise",
        "R'": "Turn the RIGHT side 90° counter-clockwise",
        "R2": "Turn the RIGHT side 180°",
        "F": "Turn the FRONT side 90° clockwise",
        "F'": "Turn the FRONT side 90° counter-clockwise",
        "F2": "Turn the FRONT side 180°",
        "B": "Turn the BACK side 90° clockwise",
        "B'": "Turn the BACK side 90° counter-clockwise",
        "B2": "Turn the BACK side 180°"
    }
    
    moves = solution_string.split()
    
    print("\n" + "="*70)
    print("🎯 RUBIK'S CUBE SOLUTION")
    print("="*70)
    print(f"\n📊 Total moves: {len(moves)}")
    print(f"🔤 Move sequence: {solution_string}\n")
    print("-"*70)
    print("📖 Step-by-Step Instructions:")
    print("-"*70)
    print("\n💡 IMPORTANT: Hold the cube with WHITE on top and GREEN facing you\n")
    print("   • TOP = White face")
    print("   • BOTTOM = Yellow face")
    print("   • FRONT = Green face (facing you)")
    print("   • BACK = Blue face (away from you)")
    print("   • RIGHT = Red face")
    print("   • LEFT = Orange face\n")
    
    for i, move in enumerate(moves, 1):
        description = move_descriptions.get(move, "Unknown move")
        print(f"  {i}) {description}")
    
    print("\n" + "="*70)
    print("✅ Follow these steps to solve your cube!")
    print("="*70 + "\n")

# ----------------------------------------------------------
# 🧩 Solve the Rubik's Cube with Auto-Calibration
# ----------------------------------------------------------
def solve_cube(captured_faces_file="captured_faces.npy"):
    """Main function to solve the Rubik's cube with auto-calibration."""
    
    try:
        # Load captured faces
        print("\n🔄 Loading captured cube data...")
        captured_faces = np.load(captured_faces_file, allow_pickle=True).item()
        
        print("✅ Cube data loaded successfully!")
        
        # Display captured faces (original orientation)
        print("\n📋 Captured Faces Configuration (As Stored):")
        print("-"*70)
        for face_name in ['white', 'orange', 'yellow', 'red', 'blue', 'green']:
            if face_name in captured_faces:
                face_array = captured_faces[face_name]
                print(f"\n{face_name.upper()} face:")
                for row in face_array:
                    print("    " + " ".join([f"{c[:3].upper():>3}" for c in row]))
        print("-"*70)
        
        # Validate cube configuration
        if not validate_cube(captured_faces):
            print("\n⚠️  Warning: Invalid color counts!")
            proceed = input("Continue with auto-calibration anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("❌ Solve cancelled.")
                return None
        
        # AUTO-CALIBRATION: Find correct rotations
        rotation_config, transformed_faces, kociemba_string = auto_calibrate(captured_faces)
        
        if rotation_config is None:
            print("\n❌ Auto-calibration failed!")
            print("\n💡 Possible reasons:")
            print("   1. Color detection errors during capture")
            print("   2. Cube was moved between face captures")
            print("   3. Some faces were captured incorrectly")
            print("\n📸 Please re-capture all faces with better lighting")
            return None
        
        # Display transformed faces
        print("\n📋 Transformed Faces (Kociemba Format):")
        print("-"*70)
        face_order = ['white', 'red', 'green', 'yellow', 'orange', 'blue']
        face_labels = ['U (Up/White)', 'R (Right/Red)', 'F (Front/Green)', 
                       'D (Down/Yellow)', 'L (Left/Orange)', 'B (Back/Blue)']
        
        for face_color, label in zip(face_order, face_labels):
            face_array = transformed_faces[face_color]
            print(f"\n{label}:")
            for row in face_array:
                print("    " + " ".join([f"{c[:3].upper():>3}" for c in row]))
        print("-"*70)
        
        print(f"\n✅ Kociemba string: {kociemba_string}")
        print(f"   Length: {len(kociemba_string)} characters")
        
        # Solve the cube
        print("\n🧠 Solving the cube using Kociemba algorithm...")
        print("   (This may take a few seconds...)")
        
        solution = kociemba.solve(kociemba_string)
        
        # Format and display solution
        format_solution(solution)
        
        # Save solution to file
        with open("solution.txt", "w") as f:
            f.write(f"Rubik's Cube Solution\n")
            f.write(f"={'='*50}\n\n")
            f.write(f"Rotation Configuration (lateral inversion + rotation):\n")
            for color in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
                f.write(f"  {color.upper():8s}: {rotation_config[color]:3d}° clockwise\n")
            f.write(f"\nMove Sequence: {solution}\n")
            f.write(f"Total Moves: {len(solution.split())}\n\n")
            f.write(f"Steps:\n")
            for i, move in enumerate(solution.split(), 1):
                f.write(f"{i}. {move}\n")
        
        print("💾 Solution saved to 'solution.txt'")
        print("💾 Rotation configuration saved for future use")
        
        return solution
        
    except FileNotFoundError:
        print(f"\n❌ Error: '{captured_faces_file}' not found!")
        print("Please run color_prediction.py first to capture the cube.")
        return None
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ----------------------------------------------------------
# 🎮 Interactive solver
# ----------------------------------------------------------
def interactive_solve():
    """Interactive mode for solving the cube."""
    
    print("\n" + "="*70)
    print("🎲 RUBIK'S CUBE SOLVER - WITH AUTO-CALIBRATION")
    print("="*70)
    print("\n✨ This solver automatically finds the correct face orientations!")
    
    while True:
        print("\nOptions:")
        print("  1. Solve cube from 'captured_faces.npy' (AUTO-CALIBRATION)")
        print("  2. Enter custom file path")
        print("  3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            solve_cube()
            break
        elif choice == "2":
            filepath = input("Enter the path to your .npy file: ").strip()
            solve_cube(filepath)
            break
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

# ----------------------------------------------------------
# 🚀 Main execution
# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        import kociemba
        print("✅ Kociemba library loaded successfully")
    except ImportError:
        print("\n❌ Error: 'kociemba' package not installed!")
        print("Please install it using: pip install kociemba")
        exit(1)
    
    interactive_solve()