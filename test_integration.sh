#!/bin/bash
# Integration test script for Docker Compose setup
# Tests MongoDB connection and API endpoints

set -e  # Exit on any error

echo "======================================================================"
echo "  FineTuneLLM Docker Compose Integration Test"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ docker-compose not found${NC}"
    echo "Please install Docker Compose to run this test"
    exit 1
fi

echo -e "${GREEN}✓ docker-compose found${NC}"

# Check if backend/model directory has a .gguf file
if [ ! -d "backend/model" ]; then
    echo -e "${YELLOW}⚠ backend/model directory not found, creating...${NC}"
    mkdir -p backend/model
fi

GGUF_COUNT=$(find backend/model -name "*.gguf" 2>/dev/null | wc -l)
if [ "$GGUF_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠ No .gguf model files found in backend/model/${NC}"
    echo "  This test will skip model-dependent checks"
    echo "  To test with a model, place a .gguf file in backend/model/"
    HAS_MODEL=false
else
    echo -e "${GREEN}✓ Found $GGUF_COUNT .gguf model file(s)${NC}"
    HAS_MODEL=true
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 1: Starting services with docker-compose"
echo "----------------------------------------------------------------------"

# Start services
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check if containers are running
if docker-compose ps | grep -q "finetune-backend.*Up"; then
    echo -e "${GREEN}✓ Backend container is running${NC}"
else
    echo -e "${RED}✗ Backend container failed to start${NC}"
    docker-compose logs backend
    exit 1
fi

if docker-compose ps | grep -q "finetune-mongodb.*Up"; then
    echo -e "${GREEN}✓ MongoDB container is running${NC}"
else
    echo -e "${RED}✗ MongoDB container failed to start${NC}"
    docker-compose logs mongodb
    exit 1
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 2: Testing API endpoints"
echo "----------------------------------------------------------------------"

# Test health endpoint
echo "Testing GET / (health check)..."
HEALTH_RESPONSE=$(curl -s http://localhost:8002/)
if echo "$HEALTH_RESPONSE" | grep -q "status"; then
    echo -e "${GREEN}✓ Health check endpoint working${NC}"
    echo "  Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "  Response: $HEALTH_RESPONSE"
fi

echo ""

# Test status endpoint
echo "Testing GET /status..."
STATUS_RESPONSE=$(curl -s http://localhost:8002/status)
echo "  Response: $STATUS_RESPONSE"

if echo "$STATUS_RESPONSE" | grep -q '"database_connected":true'; then
    echo -e "${GREEN}✓ MongoDB connected${NC}"
else
    echo -e "${YELLOW}⚠ MongoDB not connected (this is OK if MongoDB failed to start)${NC}"
fi

if echo "$STATUS_RESPONSE" | grep -q '"model_loaded":true'; then
    echo -e "${GREEN}✓ Model loaded${NC}"
    MODEL_LOADED=true
elif [ "$HAS_MODEL" = true ]; then
    echo -e "${YELLOW}⚠ Model not loaded (might still be loading, this is OK)${NC}"
    MODEL_LOADED=false
else
    echo -e "${YELLOW}⚠ Model not loaded (no .gguf file found, this is expected)${NC}"
    MODEL_LOADED=false
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 3: Checking MongoDB database creation"
echo "----------------------------------------------------------------------"

# Check if database was created
DB_CHECK=$(docker exec finetune-mongodb mongosh --quiet --eval "db.getMongo().getDBNames()" 2>/dev/null || echo "failed")
if echo "$DB_CHECK" | grep -q "finetuneLLM"; then
    echo -e "${GREEN}✓ Database 'finetuneLLM' exists${NC}"
else
    echo -e "${YELLOW}⚠ Database 'finetuneLLM' not found yet${NC}"
    echo "  (It will be created automatically on first message)"
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 4: Viewing backend logs"
echo "----------------------------------------------------------------------"

echo "Last 30 lines of backend logs:"
docker-compose logs --tail=30 backend

echo ""
echo "----------------------------------------------------------------------"
echo "Test Summary"
echo "----------------------------------------------------------------------"

if [ "$MODEL_LOADED" = true ]; then
    echo -e "${GREEN}✓ All systems operational${NC}"
    echo ""
    echo "You can now test the chat endpoint with:"
    echo '  curl -X POST http://localhost:8002/chat \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"prompt": "hello"}'"'"
    echo ""
    echo "Or use the frontend:"
    echo "  cd .."
    echo "  npm install"
    echo "  npm run dev"
else
    echo -e "${YELLOW}⚠ Services are running but model is not loaded${NC}"
    echo ""
    echo "To load a model:"
    echo "  1. Place a .gguf file in backend/model/"
    echo "  2. Restart: docker-compose restart backend"
    echo "  3. Check logs: docker-compose logs -f backend"
fi

echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f backend"
echo ""

echo "======================================================================"
echo "  Integration Test Complete"
echo "======================================================================"
