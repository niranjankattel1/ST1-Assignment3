from pathlib import Path
from .services.workflow import WorkflowService

class ConsoleApp:
    """Menu-driven console application for the workflow."""

    def __init__(self, workflow_service: "WorkflowService") -> None:
        self.workflow_service = workflow_service

    def run(self) -> None:
        """Start the menu loop until the user chooses to exit."""
        while True:
            self._print_menu()
            choice = self._get_menu_choice()

            try:
                if choice == "1":
                    self.workflow_service.show_summary()
                    print("Dataset summary complete.")
                elif choice == "2":
                    eda_performed = self._run_eda_option()
                    if eda_performed:
                        print("EDA task completed.")
                elif choice == "3":
                    results = self.workflow_service.train_model()
                    print(f"Training accuracy: {results['accuracy']:.4f}")
                    print(results["report"])
                    print("Training complete.")
                elif choice == "4":
                    image_path = self._get_image_path()
                    if image_path is None:
                        continue
                    result = self.workflow_service.predict_image(image_path)
                    print(
                        f"Predicted class: {result['predicted_class']} "
                        f"({result['confidence']:.2f}% confidence)"
                    )
                    print("Prediction complete.")
                elif choice == "5":
                    print("Exiting application.")
                    break
            except FileNotFoundError as exc:
                print(f"Error: {exc}")
            except Exception as exc:
                print(f"An error occurred: {exc}")

    @staticmethod
    def _print_menu() -> None:
        print("-----------------------------------------")
        print("Macroinvertebrate Image Analysis System")
        print("1. Show dataset summary")
        print("2. Generate EDA outputs")
        print("3. Train baseline classifier")
        print("4. Predict an image")
        print("5. Exit")
        print("-----------------------------------------")

    @staticmethod
    def _get_menu_choice() -> str:
        while True:
            choice = input("Select an option (1-5): ").strip()
            if choice in {"1", "2", "3", "4", "5"}:
                return choice
            print("Invalid option. Please enter a number from 1 to 5.")

    def _run_eda_option(self) -> bool:
        available_classes = self.workflow_service.get_available_classes()

        if not available_classes:
            print("No class folders were found for class-specific EDA.")
            self.workflow_service.generate_eda()
            return True

        while True:
            print("EDA options:")
            print("1. Full dataset EDA")
            print("2. Class-specific EDA")
            print("b. Back")
            choice = input("Select an EDA option: ").strip().lower()

            if choice == "1":
                self.workflow_service.generate_eda()
                return True
            if choice == "2":
                selected_class = self._choose_eda_class(available_classes)
                if selected_class is None:
                    return False
                self.workflow_service.generate_eda_for_class(selected_class)
                return True
            if choice in {"b", "back"}:
                return False
            print("Invalid option. Please enter 1, 2, or 'b'.")

    def _choose_eda_class(self, available_classes: list[str]) -> str | None:
        print("Available classes:")
        for index, class_name in enumerate(available_classes, start=1):
            print(f"{index}. {class_name}")

        while True:
            choice = input("Enter class number or name (or 'b' to go back): ").strip()
            if choice.lower() in {"b", "back"}:
                return None
            if choice.isdigit():
                selected_index = int(choice)
                if 1 <= selected_index <= len(available_classes):
                    return available_classes[selected_index - 1]
                print("Invalid class number. Please choose a valid index.")
                continue
            if choice in available_classes:
                return choice
            print("Invalid class name. Please choose a valid option from the list.")

    @staticmethod
    def _get_image_path() -> str | None:
        while True:
            image_path = input("Enter image path (or type 'b' to go back): ").strip()
            if image_path.lower() in {"b", "back"}:
                return None

            if not image_path:
                print("Image path cannot be empty. Please try again or type 'b' to return to the menu.")
                continue

            path = Path(image_path)
            if not path.exists() or not path.is_file():
                print("File not found. Please enter a valid image path or type 'b' to go back.")
                continue
            return str(path)


if __name__ == "__main__":
    workflow = WorkflowService()
    ConsoleApp(workflow).run()
