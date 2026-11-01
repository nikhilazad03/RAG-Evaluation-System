import sys
import subprocess
from pathlib import Path

# Add RAG1 to path
sys.path.append(str(Path(__file__).resolve().parent / "RAG1"))
from agent import ask_question

def interactive_chat():
    print("\n--- Penguin RAG Assistant (Type 'exit' or 'q' to quit) ---\n")
    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "q", "quit"]:
                print("Exiting.")
                break

            result = ask_question(query)
            print("\nAnswer:", result.get("answer"))
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def launch_dashboard():
    print("Launching Streamlit Evaluation Dashboard...")
    subprocess.run(["streamlit", "run", "dashboard.py"])

def main():
    print("================================")
    print(" Penguin RAG & Evaluation System ")
    print("================================")
    print("1. Interactive Chat (Test live RAG)")
    print("2. Launch Streamlit Comparison Dashboard")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()
    if choice == "1":
        interactive_chat()
    elif choice == "2":
        launch_dashboard()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()