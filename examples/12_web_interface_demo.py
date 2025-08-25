"""
Web Interface Demo for AI Tutorial

This example demonstrates the interactive web dashboard for the AI Tutorial,
providing a browser-based interface for learning AI and ML concepts.
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_interface.dashboard_app import create_dashboard_app


def main():
    """Main demo function"""
    print("🌐 AI TUTORIAL WEB INTERFACE DEMONSTRATION")
    print("=" * 60)
    print("This demo launches an interactive web dashboard for the")
    print("AI Tutorial, accessible through your web browser.")
    print()
    
    try:
        # Create dashboard app
        print("🚀 Starting AI Tutorial Web Dashboard...")
        dashboard = create_dashboard_app(port=8080, host='localhost', debug=False)
        
        print(f"📊 Dashboard Features:")
        print(f"  ✅ Interactive example runner")
        print(f"  ✅ Tutorial browser and progress tracking")
        print(f"  ✅ AI development tools")
        print(f"  ✅ Real-time visualizations")
        print(f"  ✅ Achievement system")
        
        print(f"\n🌐 Dashboard will be available at:")
        print(f"   http://localhost:8080")
        print(f"\n📖 Features to explore:")
        print(f"  • Run AI examples interactively")
        print(f"  • Browse structured tutorials")
        print(f"  • Use advanced AI development tools")
        print(f"  • Track your learning progress")
        print(f"  • View interactive visualizations")
        
        print(f"\n🚀 Opening dashboard in your default browser...")
        
        # Start dashboard in a separate thread
        dashboard_thread = threading.Thread(target=dashboard.run)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Try to open browser
        try:
            webbrowser.open('http://localhost:8080')
            print("✅ Browser opened successfully!")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("   Please manually navigate to http://localhost:8080")
        
        print(f"\n🎮 How to use the dashboard:")
        print(f"  1. 📚 Start with 'Examples' to run AI code interactively")
        print(f"  2. 🛠️  Try 'AI Tools' for advanced development features")
        print(f"  3. 📊 Check 'Progress' to track your learning journey")
        print(f"  4. 📖 Browse 'Tutorials' for structured learning paths")
        
        print(f"\n⭐ Key Features:")
        print(f"  • Real-time code execution")
        print(f"  • Interactive visualizations")
        print(f"  • Progress tracking and achievements")
        print(f"  • Model evaluation dashboards")
        print(f"  • Training progress visualization")
        print(f"  • Model interpretability tools")
        print(f"  • Hyperparameter optimization")
        
        print(f"\n🛑 Press Ctrl+C to stop the dashboard server")
        print("=" * 60)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping dashboard server...")
            print("✅ Dashboard stopped successfully!")
    
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n✅ Web interface demonstration completed!")


if __name__ == "__main__":
    main()