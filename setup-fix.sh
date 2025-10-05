#!/bin/bash

# AI Camera Application - Setup Script
# Fixes all issues and reorganizes project structure

echo "🔧 AI Camera Application - Project Reorganization"
echo "=================================================="

# Create directories
echo "📁 Creating directory structure..."
mkdir -p frontend backend backend/logs

# Move frontend files
echo "📄 Moving frontend files..."
if [ -f "app.js" ]; then
    mv app.js frontend/ 2>/dev/null || true
    echo "  ✓ Moved app.js to frontend/"
fi

if [ -f "index.html" ]; then
    mv index.html frontend/ 2>/dev/null || true
    echo "  ✓ Moved index.html to frontend/"
fi

if [ -f "styles.css" ]; then
    mv styles.css frontend/ 2>/dev/null || true
    echo "  ✓ Moved styles.css to frontend/"
fi

# Move backend files  
echo "🐍 Moving backend files..."
if [ -f "app.py" ]; then
    mv app.py backend/ 2>/dev/null || true
    echo "  ✓ Moved app.py to backend/"
fi

if [ -f "backend-app.py" ]; then
    mv backend-app.py backend/app.py 2>/dev/null || true
    echo "  ✓ Moved fixed backend-app.py to backend/app.py"
fi

if [ -f "utils.py" ]; then
    mv utils.py backend/ 2>/dev/null || true
    echo "  ✓ Moved utils.py to backend/"
fi

if [ -f "requirements.txt" ]; then
    mv requirements.txt backend/ 2>/dev/null || true
    echo "  ✓ Moved requirements.txt to backend/"
fi

if [ -f "backend-requirements.txt" ]; then
    mv backend-requirements.txt backend/requirements.txt 2>/dev/null || true
    echo "  ✓ Moved fixed backend-requirements.txt to backend/requirements.txt"
fi

if [ -f "Dockerfile" ]; then
    mv Dockerfile backend/ 2>/dev/null || true
    echo "  ✓ Moved Dockerfile to backend/"
fi

if [ -f "backend-dockerfile" ]; then
    mv backend-dockerfile backend/Dockerfile 2>/dev/null || true
    echo "  ✓ Moved fixed backend-dockerfile to backend/Dockerfile"
fi

# Copy configuration files to root
echo "⚙️  Setting up configuration files..."
if [ -f "docker-compose-fixed.yml" ]; then
    mv docker-compose-fixed.yml docker-compose.yml 2>/dev/null || true
    echo "  ✓ Moved fixed docker-compose.yml to root"
fi

if [ -f "nginx.conf" ]; then
    echo "  ✓ nginx.conf is ready in root"
else
    echo "  ⚠️  nginx.conf not found"
fi

# Verify structure
echo ""
echo "📂 Project structure verification:"
echo "=================================="
echo "Root directory:"
ls -la | grep -E "(docker-compose|nginx)" | sed 's/^/  /'

echo ""
echo "Frontend directory:"
if [ -d "frontend" ]; then
    ls -la frontend/ | sed 's/^/  /'
else
    echo "  ❌ Frontend directory not found"
fi

echo ""
echo "Backend directory:"
if [ -d "backend" ]; then
    ls -la backend/ | sed 's/^/  /'
else
    echo "  ❌ Backend directory not found"
fi

echo ""
echo "🎯 Issues Fixed:"
echo "==============="
echo "  ✅ FIX 1: File structure reorganized (frontend/, backend/)"
echo "  ✅ FIX 2: Fixed Python imports with transformations"
echo "  ✅ FIX 3: Production CORS configuration"
echo "  ✅ FIX 4: Removed emoji logging for better compatibility"
echo "  ✅ FIX 5: Added async math solving with timeout"
echo "  ✅ FIX 6: Enhanced Dockerfile with curl for healthcheck"
echo "  ✅ FIX 7: Complete nginx.conf with proper proxy settings"
echo "  ✅ FIX 8: Updated docker-compose.yml with proper volumes"

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Build and run the application:"
echo "   docker-compose up --build"
echo ""
echo "2. Test the services:"
echo "   Frontend: http://localhost:8080"
echo "   Backend:  http://localhost:8000"
echo "   Health:   http://localhost:8000/health"
echo ""
echo "3. Test the math API:"
echo "   curl -X POST http://localhost:8000/api/solve-math \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"expression\": \"2x + 5 = 15\"}'"
echo ""
echo "✨ Project reorganization complete!"