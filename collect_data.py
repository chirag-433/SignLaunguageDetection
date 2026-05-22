import os
import cv2
import shutil

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100

labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}


def get_existing_classes():
    """Return a dict of class_index -> letter for classes that already have data."""
    existing = {}
    for idx, letter in labels_dict.items():
        class_dir = os.path.join(DATA_DIR, str(idx))
        if os.path.isdir(class_dir) and len(os.listdir(class_dir)) > 0:
            existing[idx] = letter
    return existing


def collect_for_class(cap, class_idx, letter):
    """Collect dataset_size images for a single class."""
    class_dir = os.path.join(DATA_DIR, str(class_idx))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'\nCollecting data for class {class_idx} (Letter: {letter})')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return
        cv2.putText(frame, f'Letter: {letter}. Ready? Press "Q" !', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return
        cv2.putText(frame, f'Letter: {letter} - {counter + 1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

    print(f'Done collecting {dataset_size} images for letter: {letter}')


def show_menu():
    """Display the main menu and return the user's choice."""
    existing = get_existing_classes()

    print("\n" + "=" * 50)
    print("     HAND SIGN DATA COLLECTION TOOL")
    print("=" * 50)

    if existing:
        print(f"\nExisting data found for {len(existing)} class(es):")
        for idx in sorted(existing.keys()):
            class_dir = os.path.join(DATA_DIR, str(idx))
            num_images = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
            print(f"  Class {idx} -> Letter '{existing[idx]}' ({num_images} images)")
    else:
        print("\nNo existing data found.")

    missing = {idx: letter for idx, letter in labels_dict.items() if idx not in existing}
    if missing:
        print(f"\nMissing classes ({len(missing)}):")
        for idx in sorted(missing.keys()):
            print(f"  Class {idx} -> Letter '{missing[idx]}'")

    print("\n--- OPTIONS ---")
    print("1. Erase ALL data and recollect everything from scratch")
    print("2. Edit a specific letter (re-collect its images)")
    print("3. Add missing letter(s)")
    print("4. Collect ONLY missing letters (skip existing)")
    print("5. Exit")
    print()

    choice = input("Enter your choice (1-5): ").strip()
    return choice


def main():
    while True:
        choice = show_menu()

        if choice == '1':
            # Erase all and start over
            confirm = input("WARNING: This will DELETE all existing images. Type 'yes' to confirm: ").strip().lower()
            if confirm != 'yes':
                print("Cancelled.")
                continue

            # Delete all class directories
            for idx in labels_dict:
                class_dir = os.path.join(DATA_DIR, str(idx))
                if os.path.isdir(class_dir):
                    shutil.rmtree(class_dir)
                    print(f"  Deleted: {class_dir}")

            # Collect for all classes
            cap = cv2.VideoCapture(0)
            for idx in sorted(labels_dict.keys()):
                collect_for_class(cap, idx, labels_dict[idx])
            cap.release()
            cv2.destroyAllWindows()
            print("\nAll classes collected successfully!")

        elif choice == '2':
            # Edit a specific letter
            existing = get_existing_classes()
            all_letters = {idx: letter for idx, letter in labels_dict.items()}

            print("\nAvailable letters to edit:")
            for idx in sorted(all_letters.keys()):
                status = "(has data)" if idx in existing else "(no data)"
                print(f"  {idx}: {all_letters[idx]} {status}")

            try:
                selected = input("\nEnter the class number to re-collect (e.g. 0 for A): ").strip()
                selected = int(selected)
            except ValueError:
                print("Invalid input.")
                continue

            if selected not in labels_dict:
                print(f"Class {selected} does not exist in labels_dict.")
                continue

            # Clear old data for this class
            class_dir = os.path.join(DATA_DIR, str(selected))
            if os.path.isdir(class_dir):
                shutil.rmtree(class_dir)
                print(f"  Cleared old data for class {selected} ({labels_dict[selected]})")

            cap = cv2.VideoCapture(0)
            collect_for_class(cap, selected, labels_dict[selected])
            cap.release()
            cv2.destroyAllWindows()

        elif choice == '3':
            # Add a missing / new letter
            print("\nCurrent labels_dict:", labels_dict)
            try:
                new_idx = input("Enter class number for the new letter (e.g. 7): ").strip()
                new_idx = int(new_idx)
            except ValueError:
                print("Invalid input.")
                continue

            if new_idx in labels_dict:
                print(f"Class {new_idx} already exists as '{labels_dict[new_idx]}'. Use option 2 to edit it.")
                continue

            new_letter = input("Enter the letter/label for this class (e.g. G): ").strip().upper()
            if not new_letter:
                print("Invalid letter.")
                continue

            labels_dict[new_idx] = new_letter
            print(f"Added class {new_idx} -> '{new_letter}'")

            cap = cv2.VideoCapture(0)
            collect_for_class(cap, new_idx, new_letter)
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDone! Remember to update labels_dict in this file to include {new_idx}: '{new_letter}'")

        elif choice == '4':
            # Collect only missing letters
            existing = get_existing_classes()
            missing = {idx: letter for idx, letter in labels_dict.items() if idx not in existing}

            if not missing:
                print("\nNo missing classes! All letters already have data.")
                continue

            print(f"\nWill collect data for {len(missing)} missing class(es):")
            for idx in sorted(missing.keys()):
                print(f"  Class {idx} -> Letter '{missing[idx]}'")

            cap = cv2.VideoCapture(0)
            for idx in sorted(missing.keys()):
                collect_for_class(cap, idx, missing[idx])
            cap.release()
            cv2.destroyAllWindows()
            print("\nAll missing classes collected successfully!")

        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == '__main__':
    main()
