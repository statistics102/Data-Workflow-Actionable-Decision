# launch_dashboard.py
import subprocess
import sys
import os
import webbrowser
import threading
import time

def install_requirements():
    """Install required packages for the dashboard"""
    print("📦 Installing dashboard dependencies...")
    requirements = [
        'plotly>=5.0.0',
        'dash>=2.0.0',
        'dash-bootstrap-components>=1.0.0'
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def check_data_files():
    """Ensure data files exist"""
    if not os.path.exists('data/cleaned_analysis_data.csv'):
        print("🔄 Generating sample data...")
        from generate_sample_data import create_sample_data
        from complete_analysis_pipeline import DataAnalysisPipeline
        
        create_sample_data()
        pipeline = DataAnalysisPipeline()
        pipeline.load_data().data_cleaning().exploratory_analysis()
        print("✅ Data preparation completed")

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://127.0.0.1:8050')

def main():
    print("🚀 Professional Live Dashboard Launcher")
    print("=" * 50)
    
    # Check and install requirements
    try:
        import plotly
        import dash
        print("✅ All dashboard dependencies are already installed")
    except ImportError:
        print("❌ Some dependencies missing. Installing now...")
        install_requirements()
    
    # Check data files
    check_data_files()
    
    # Launch dashboard in a separate thread and open browser
    print("\n🎯 Starting Live Professional Dashboard...")
    print("🌐 Dashboard will open at: http://127.0.0.1:8050")
    print("🔄 Features:")
    print("   - Real-time updates every 10 seconds")
    print("   - Interactive filters and controls")
    print("   - Professional dark theme")
    print("   - Customer segmentation analysis")
    print("   - Feature importance ranking")
    print("   - 3D clustering visualization")
    print("\n⏹️  Press Ctrl+C to stop the dashboard")
    
    # Open browser in separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run dashboard
    from live_dashboard import app
    app.run(debug=True, host='127.0.0.1', port=8050)

if __name__ == '__main__':
    main()