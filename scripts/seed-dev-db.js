/**
 * Development Database Seeding Script
 * 
 * This script populates the development database with initial data for testing.
 */

require('dotenv').config({ path: '.env.development' });
const mongoose = require('mongoose');
const { exit } = require('process');

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => {
  console.error('MongoDB connection error:', err);
  exit(1);
});

// Import models (adjust paths as needed)
let User, Trade, Market;
try {
  User = require('../models/User');
  Trade = require('../models/Trade');
  Market = require('../models/Market');
} catch (error) {
  console.error('Failed to import models:', error);
  console.log('Make sure your models are properly defined.');
  exit(1);
}

// Sample data
const users = [
  {
    username: 'admin',
    email: 'admin@example.com',
    password: 'admin123', // This will be hashed by the model's pre-save hook
    role: 'admin',
    firstName: 'Admin',
    lastName: 'User'
  },
  {
    username: 'trader1',
    email: 'trader1@example.com',
    password: 'trader123',
    role: 'user',
    firstName: 'John',
    lastName: 'Doe'
  },
  {
    username: 'trader2',
    email: 'trader2@example.com',
    password: 'trader456',
    role: 'user',
    firstName: 'Jane',
    lastName: 'Smith'
  }
];

const markets = [
  {
    name: 'BTC-USD',
    baseCurrency: 'BTC',
    quoteCurrency: 'USD',
    minOrderSize: 0.001,
    active: true
  },
  {
    name: 'ETH-USD',
    baseCurrency: 'ETH',
    quoteCurrency: 'USD',
    minOrderSize: 0.01,
    active: true
  },
  {
    name: 'XRP-USD',
    baseCurrency: 'XRP',
    quoteCurrency: 'USD',
    minOrderSize: 1,
    active: true
  }
];

// Seed function
async function seedDatabase() {
  try {
    // Clear existing data
    console.log('Clearing existing data...');
    await User.deleteMany({});
    await Market.deleteMany({});
    await Trade.deleteMany({});

    // Insert users
    console.log('Inserting users...');
    const createdUsers = await User.create(users);

    // Insert markets
    console.log('Inserting markets...');
    const createdMarkets = await Market.create(markets);

    // Insert sample trades
    console.log('Inserting sample trades...');
    const trades = [];

    // Create some sample trades
    for (let i = 0; i < 50; i++) {
      const randomUser = createdUsers[Math.floor(Math.random() * createdUsers.length)];
      const randomMarket = createdMarkets[Math.floor(Math.random() * createdMarkets.length)];

      const trade = {
        user: randomUser._id,
        market: randomMarket._id,
        type: Math.random() > 0.5 ? 'buy' : 'sell',
        price: parseFloat((Math.random() * 1000 + 10000).toFixed(2)),
        amount: parseFloat((Math.random() * 10).toFixed(6)),
        status: 'completed',
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 30 * 24 * 60 * 60 * 1000))
      };

      trade.total = trade.price * trade.amount;
      trades.push(trade);
    }

    await Trade.create(trades);

    console.log('Database seeding completed successfully!');
    console.log(`Created ${createdUsers.length} users`);
    console.log(`Created ${createdMarkets.length} markets`);
    console.log(`Created ${trades.length} trades`);

    mongoose.connection.close();
  } catch (error) {
    console.error('Seeding error:', error);
    mongoose.connection.close();
    process.exit(1);
  }
}

// Run the seed function
seedDatabase();
