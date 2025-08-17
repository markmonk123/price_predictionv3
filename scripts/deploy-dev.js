/**
 * Development Deployment Script
 * 
 * This script handles the deployment of the application to a development environment.
 * It copies build files to the appropriate locations and restarts the necessary services.
 */

require('dotenv').config({ path: '.env.development' });
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const config = {
  buildDir: path.join(__dirname, '../client/build'),
  deployDir: process.env.DEPLOY_DIR || '/var/www/bitcoin-fix-dev',
  backupDir: '/var/www/backups/bitcoin-fix-dev',
  serviceNames: ['bitcoin-fix-dev']
};

console.log('Starting development deployment...');

try {
  // 1. Create backup of current deployment
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = `${config.backupDir}/${timestamp}`;

  console.log(`Creating backup at ${backupPath}...`);
  fs.mkdirSync(backupPath, { recursive: true });

  if (fs.existsSync(config.deployDir)) {
    execSync(`cp -r ${config.deployDir}/* ${backupPath}`);
  }

  // 2. Ensure deploy directory exists
  console.log(`Ensuring deploy directory exists: ${config.deployDir}`);
  fs.mkdirSync(config.deployDir, { recursive: true });

  // 3. Copy build files to deployment directory
  console.log('Copying build files to deployment directory...');
  execSync(`cp -r ${config.buildDir}/* ${config.deployDir}/public/`);

  // 4. Copy server files to deployment directory
  console.log('Copying server files to deployment directory...');
  const serverFiles = [
    'server.js',
    'package.json',
    'package-lock.json',
    '.env.development'
  ];

  const serverDirs = [
    'models',
    'routes',
    'controllers',
    'middleware',
    'utils'
  ];

  serverFiles.forEach(file => {
    if (fs.existsSync(path.join(__dirname, '..', file))) {
      fs.copyFileSync(
        path.join(__dirname, '..', file),
        path.join(config.deployDir, file)
      );
    }
  });

  serverDirs.forEach(dir => {
    const srcDir = path.join(__dirname, '..', dir);
    const destDir = path.join(config.deployDir, dir);

    if (fs.existsSync(srcDir)) {
      fs.mkdirSync(destDir, { recursive: true });
      execSync(`cp -r ${srcDir}/* ${destDir}`);
    }
  });

  // 5. Install dependencies in deployment directory
  console.log('Installing dependencies in deployment directory...');
  execSync('npm install --production', { cwd: config.deployDir });

  // 6. Restart services
  console.log('Restarting services...');
  config.serviceNames.forEach(service => {
    try {
      execSync(`sudo systemctl restart ${service}`);
      console.log(`Service ${service} restarted successfully.`);
    } catch (error) {
      console.error(`Failed to restart service ${service}:`, error.message);
    }
  });

  console.log('Deployment completed successfully!');
} catch (error) {
  console.error('Deployment failed:', error);
  process.exit(1);
}
