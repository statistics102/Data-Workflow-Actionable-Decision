#!/bin/bash
# setup_git.sh - Complete Git setup for Windows/MINGW64

echo "🚀 Setting up Git repository..."
echo "=========================================="

# Change to project directory
cd "C:/Users/user/Documents/Data-Workflow-Actionable-Decision" || {
    echo "❌ Error: Cannot navigate to project directory"
    exit 1
}

echo "📁 Current directory: $(pwd)"

# Clean up any Git issues
echo "🔧 Cleaning up Git issues..."
rm -f .git/index.lock
rm -f .git/HEAD.lock

# Initialize Git if needed
if [ ! -d ".git" ]; then
    echo "🔄 Initializing new Git repository..."
    git init
fi

# Check Git status
echo "📊 Git status:"
git status

# Add all files
echo "📦 Adding files to Git..."
git add .

# Check what will be committed
echo "📋 Files to be committed:"
git status --short

# Commit changes
echo "💾 Committing changes..."
git commit -m "🚀 Initial commit: Data Workflow Actionable Decision System

- Complete data analysis workflow
- Interactive dashboards and visualizations
- Decision tree intelligence engine
- Business rule extraction
- Professional documentation
- Easy deployment scripts"

echo "✅ Local Git repository setup complete!"

# If you want to connect to GitHub (uncomment and modify)
echo "🌐 To connect to GitHub, run:"
echo "   git remote add origin https://github.com/statistics102/Data-Workflow-Actionable-Decision.git"
echo "   git branch -M main"
echo "   git push -u origin main"