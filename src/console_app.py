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
                elif choice == "2":
                    self.workflow_service.generate_eda()
                elif choice == "3":
                    self.workflow_service.train_model()
                elif choice == "4":
                    image_path = self._get_image_path()
                    if image_path is None:
                        continue
                    self.workflow_service.predict_image(image_path)
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
