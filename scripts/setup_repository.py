# setup_repository.py
import os
import shutil
import subprocess
import sys

def create_folder_structure():
    """Create the complete folder structure"""
    print("📁 Creating folder structure...")
    
    folders = [
        'data',
        'scripts', 
        'outputs',
        'docs',
        'config',
        'launchers'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created folder: {folder}/")

def create_required_files():
    """Create all required files with proper content"""
    print("\n📄 Creating required files...")
    
    # Create requirements.txt
    requirements_content = """# Core Data Science
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Machine Learning
scikit-learn>=1.1.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Interactive Dashboards
dash>=2.7.0
dash-bootstrap-components>=1.3.0
dash-core-components>=2.0.0
dash-html-components>=2.0.0
dash-table>=5.0.0

# Development & Utilities
jupyter>=1.0.0
notebook>=6.5.0
ipython>=8.5.0
"""
    
    with open('config/requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("✅ Created: config/requirements.txt")
    
    # Create empty data files if they don't exist
    data_files = ['data/raw_customer_data.csv', 'data/cleaned_analysis_data.csv']
    for file in data_files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write("# Sample data file - replace with your data\n")
            print(f"✅ Created: {file}")

def fix_git_issues():
    """Fix any Git-related issues"""
    print("\n🔧 Fixing Git issues...")
    
    try:
        # Remove any existing lock files
        lock_files = [
            os.path.expanduser('~/.git/index.lock'),
            '.git/index.lock',
            '.git/HEAD.lock'
        ]
        
        for lock_file in lock_files:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                print(f"✅ Removed lock file: {lock_file}")
        
        # Reinitialize Git if needed
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True)
            print("✅ Initialized Git repository")
        
        print("✅ Git issues resolved")
        
    except Exception as e:
        print(f"⚠️  Git issue: {e}")

def verify_setup():
    """Verify the setup is complete"""
    print("\n🔍 Verifying setup...")
    
    required_folders = ['data', 'scripts', 'outputs', 'docs', 'config', 'launchers']
    required_files = [
        'config/requirements.txt',
        'data/raw_customer_data.csv', 
        'data/cleaned_analysis_data.csv'
    ]
    
    all_good = True
    
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✅ Folder exists: {folder}/")
        else:
            print(f"❌ Missing folder: {folder}/")
            all_good = False
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ File exists: {file}")
        else:
            print(f"❌ Missing file: {file}")
            all_good = False
    
    return all_good

def main():
    print("🚀 Data Workflow Actionable Decision - Complete Setup")
    print("=" * 60)
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Fix Git issues first
    fix_git_issues()
    
    # Create folder structure
    create_folder_structure()
    
    # Create required files
    create_required_files()
    
    # Verify setup
    if verify_setup():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Add your Python scripts to the 'scripts/' folder")
        print("2. Add your data files to the 'data/' folder") 
        print("3. Run: git add .")
        print("4. Run: git commit -m 'Initial commit'")
        print("5. Run: git push origin main")
    else:
        print("\n❌ Setup incomplete. Please check missing items.")

if __name__ == '__main__':
    main()