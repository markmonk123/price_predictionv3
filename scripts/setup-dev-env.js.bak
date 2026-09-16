/**
 * Development Environment Setup Script
 * 
 * This script sets up the development environment for the Bitcoin FIX Trading Platform.
 * It creates necessary directories, configuration files, and environment variables.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('Setting up development environment...');

try {
  // 1. Create necessary directories
  const dirs = [
    'logs',
    'data',
    'data/db',
    'uploads',
    'config'
  ];

  dirs.forEach(dir => {
    const dirPath = path.join(__dirname, '..', dir);
    if (!fs.existsSync(dirPath)) {
      console.log(`Creating directory: ${dir}`);
      fs.mkdirSync(dirPath, { recursive: true });
    }
  });

  // 2. Create development environment file if it doesn't exist
  const envDevPath = path.join(__dirname, '..', '.env.development');
  if (!fs.existsSync(envDevPath)) {
    console.log('Creating .env.development file...');
    const envContent = `# Development Environment Configuration

# Server Configuration
NODE_ENV=development
PORT=4000
API_PREFIX=/api/v1

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/bitcoin_fix_dev

# JWT Secret
JWT_SECRET=your_jwt_secret_here
JWT_EXPIRES_IN=1d

# Coinbase API
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_api_secret

# FIX Protocol
FIX_SENDER_COMP_ID=BITCOIN_FIX_DEV
FIX_TARGET_COMP_ID=EXCHANGE
FIX_HOST=localhost
FIX_PORT=9878

# Deployment
DEPLOY_DIR=/var/www/bitcoin-fix-dev
`;

    fs.writeFileSync(envDevPath, envContent);
  }

  // 3. Create docker-compose.dev.yml if it doesn't exist
  const dockerComposePath = path.join(__dirname, '..', 'docker-compose.dev.yml');
  if (!fs.existsSync(dockerComposePath)) {
    console.log('Creating docker-compose.dev.yml file...');
    const dockerComposeContent = `version: '3.8'

services:
  # MongoDB service
  mongo:
    image: mongo:latest
    container_name: bitcoin-fix-mongo-dev
    ports:
      - "27017:27017"
    volumes:
      - ./data/db:/data/db
    restart: unless-stopped
    environment:
      - MONGO_INITDB_DATABASE=bitcoin_fix_dev

  # MongoDB UI (optional)
  mongo-express:
    image: mongo-express:latest
    container_name: bitcoin-fix-mongo-express-dev
    ports:
      - "8081:8081"
    environment:
      - ME_CONFIG_MONGODB_SERVER=mongo
    depends_on:
      - mongo
    restart: unless-stopped

  # Redis for caching
  redis:
    image: redis:alpine
    container_name: bitcoin-fix-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    restart: unless-stopped
`;

    fs.writeFileSync(dockerComposePath, dockerComposeContent);
  }

  // 4. Create client .env.development if it doesn't exist
  const clientEnvPath = path.join(__dirname, '..', 'client', '.env.development');
  if (!fs.existsSync(clientEnvPath)) {
    console.log('Creating client .env.development file...');

    // Make sure client directory exists
    const clientDir = path.join(__dirname, '..', 'client');
    if (!fs.existsSync(clientDir)) {
      fs.mkdirSync(clientDir, { recursive: true });
    }

    const clientEnvContent = `# Client Development Environment Configuration

REACT_APP_API_URL=http://localhost:4000/api/v1
REACT_APP_WS_URL=ws://localhost:4000
REACT_APP_ENV=development
`;

    fs.writeFileSync(clientEnvPath, clientEnvContent);
  }

  // 5. Install development dependencies
  console.log('Installing development dependencies...');
  execSync('npm install', { stdio: 'inherit' });

  // 6. Set up Git hooks (if not already set up)
  try {
    console.log('Setting up Git hooks...');

    // Check if .git exists
    if (fs.existsSync(path.join(__dirname, '..', '.git'))) {
      // Create .husky directory if it doesn't exist
      const huskyDir = path.join(__dirname, '..', '.husky');
      if (!fs.existsSync(huskyDir)) {
        fs.mkdirSync(huskyDir, { recursive: true });

        // Initialize husky
        execSync('npx husky install', { stdio: 'inherit' });

        // Add pre-commit hook
        execSync('npx husky add .husky/pre-commit "npx lint-staged"', { stdio: 'inherit' });
      }
    } else {
      console.log('No Git repository found, skipping Git hooks setup.');
    }
  } catch (error) {
    console.warn('Failed to set up Git hooks:', error.message);
  }

  console.log('\nDevelopment environment setup completed successfully!');
  console.log('\nNext steps:');
  console.log('1. Update the .env.development file with your actual credentials');
  console.log('2. Start the development services with: npm run docker:dev');
  console.log('3. Start the development server with: npm run dev-full');

} catch (error) {
  console.error('Setup failed:', error);
  process.exit(1);
}
