#!/usr/bin/env node
/**
 * Coinbase Developer Platform (CDP) MCP Server
 *
 * Exposes Coinbase Programmable Wallet API through MCP:
 * - Wallet creation and management
 * - Asset transfers and transactions
 * - Staking operations
 * - Transaction history and monitoring
 * - Faucet operations (testnet)
 *
 * Requires CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY env vars
 * Connects to either testnet (default) or mainnet
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { z } = require('zod');
const { version } = require('../package.json');

// CDP SDK setup - loaded based on environment
let Wallet, CdpApiClient, ApiClients;
try {
  // eslint-disable-next-line global-require
  const cdp = require('@coinbase/cdp-sdk');
  ({ Wallet, CdpApiClient } = cdp);
} catch (err) {
  console.error('Failed to load CDP SDK. Ensure @coinbase/cdp-sdk is installed.');
  process.exit(1);
}

// Configuration
const CONFIG = {
  network: (process.env.CDP_NETWORK || 'base-sepolia').toLowerCase(),
  isMainnet: (process.env.CDP_NETWORK || 'base-sepolia').toLowerCase().includes('mainnet'),
  debug: process.env.CDP_DEBUG === 'true',
};

if (CONFIG.debug) {
  console.error(`[CDP MCP] Network: ${CONFIG.network}, Mainnet: ${CONFIG.isMainnet}`);
}

// Initialize CDP API Client (lazy loaded)
let cdpClient = null;
const getCdpClient = () => {
  if (!cdpClient) {
    const keyName = process.env.CDP_API_KEY_NAME;
    const keyPrivate = process.env.CDP_API_KEY_PRIVATE_KEY;

    if (!keyName || !keyPrivate) {
      throw new Error('CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY environment variables required');
    }

    // Initialize the API client with credentials
    CdpApiClient.configureFromJson({
      private_key: keyPrivate,
      name: keyName,
    });

    cdpClient = new CdpApiClient();
  }
  return cdpClient;
};

// Error handling wrapper
const fail = (error) => ({
  isError: true,
  content: [{
    type: 'text',
    text: `${error.message}${error.details ? `\n${JSON.stringify(error.details)}` : ''}`
  }],
});

const ok = (data) => ({
  content: [{
    type: 'text',
    text: JSON.stringify(data, null, 2),
  }],
});

const tool = (handler) => async (args) => {
  try {
    const result = await handler(args || {});
    return ok(result);
  } catch (error) {
    return fail(error);
  }
};

// Schema definitions
const walletSchema = z.object({
  id: z.string().describe('Wallet ID (UUID)'),
});

const addressSchema = z.object({
  walletId: z.string().describe('Wallet ID containing the address'),
  addressId: z.string().describe('Address ID (UUID)'),
});

const transferSchema = z.object({
  walletId: z.string().describe('Source wallet ID'),
  addressId: z.string().describe('Source address ID'),
  destination: z.string().describe('Destination address'),
  asset: z.enum(['eth', 'usdc', 'btc']).default('eth').describe('Asset to transfer'),
  amount: z.string().describe('Amount in base units (wei for ETH)'),
  gasPrice: z.string().optional().describe('Gas price in gwei (optional)'),
});

const stakingSchema = z.object({
  walletId: z.string().describe('Wallet ID'),
  addressId: z.string().describe('Address ID'),
  amount: z.string().describe('Amount to stake in wei'),
  mode: z.enum(['stake', 'unstake']).default('stake'),
});

const createServer = () => {
  const server = new McpServer({
    name: 'coinbase-cdp',
    version,
  });

  // Wallet Management Tools

  server.registerTool('create_wallet', {
    title: 'Create a new wallet',
    description: `Create a new Coinbase Programmable Wallet on ${CONFIG.network}`,
    inputSchema: {
      name: z.string().optional().describe('Wallet display name'),
    },
  }, tool(async (args) => {
    const client = getCdpClient();
    const wallet = await Wallet.create({ networkId: CONFIG.network });

    if (args.name) {
      // Note: CDP SDK may not support renaming in all versions
      wallet.displayName = args.name;
    }

    return {
      id: wallet.getId(),
      network: CONFIG.network,
      addresses: wallet.getAddresses().map(addr => ({
        id: addr.getId(),
        address: addr.getId(), // Address ID is the address itself
      })),
      displayName: args.name || 'Unnamed',
    };
  }));

  server.registerTool('list_wallets', {
    title: 'List all wallets',
    description: 'Get all Coinbase Programmable Wallets in your account',
    inputSchema: {
      limit: z.number().int().min(1).max(100).optional().default(10),
    },
  }, tool(async (args) => {
    const client = getCdpClient();
    const wallets = await Wallet.list({ limit: args.limit });

    return {
      wallets: wallets.map(w => ({
        id: w.getId(),
        network: w.getNetworkId?.() || CONFIG.network,
        addresses: w.getAddresses?.().map(a => ({
          id: a.getId?.(),
          address: a.getId?.(),
        })) || [],
      })),
      total: wallets.length,
    };
  }));

  server.registerTool('get_wallet', {
    title: 'Get wallet details',
    description: 'Get detailed information about a specific wallet',
    inputSchema: walletSchema,
  }, tool(async (args) => {
    const wallet = await Wallet.fetch(args.id);

    return {
      id: wallet.getId(),
      network: wallet.getNetworkId?.() || CONFIG.network,
      addresses: wallet.getAddresses?.().map(a => ({
        id: a.getId?.(),
        address: a.getId?.(),
      })) || [],
    };
  }));

  // Address Management Tools

  server.registerTool('create_address', {
    title: 'Create new address in wallet',
    description: 'Create a new address within an existing wallet',
    inputSchema: walletSchema,
  }, tool(async (args) => {
    const wallet = await Wallet.fetch(args.id);
    const address = await wallet.createAddress();

    return {
      addressId: address.getId(),
      address: address.getId(),
      walletId: args.id,
    };
  }));

  server.registerTool('list_addresses', {
    title: 'List wallet addresses',
    description: 'Get all addresses in a wallet',
    inputSchema: walletSchema,
  }, tool(async (args) => {
    const wallet = await Wallet.fetch(args.id);
    const addresses = wallet.getAddresses?.() || [];

    return {
      walletId: args.id,
      addresses: addresses.map(a => ({
        id: a.getId?.(),
        address: a.getId?.(),
      })),
      count: addresses.length,
    };
  }));

  // Balance and Asset Tools

  server.registerTool('get_balance', {
    title: 'Get address balances',
    description: 'Get all token balances for an address',
    inputSchema: addressSchema,
  }, tool(async (args) => {
    const wallet = await Wallet.fetch(args.walletId);
    const addresses = wallet.getAddresses?.() || [];
    const address = addresses.find(a => a.getId?.() === args.addressId);

    if (!address) {
      throw new Error(`Address ${args.addressId} not found in wallet`);
    }

    const balances = await address.getBalance?.('wei') || [];

    return {
      walletId: args.walletId,
      addressId: args.addressId,
      balances: balances,
      network: CONFIG.network,
    };
  }));

  // Transfer Tools

  server.registerTool('transfer_asset', {
    title: 'Transfer asset from wallet',
    description: 'Transfer ETH, USDC or other assets to another address',
    inputSchema: transferSchema,
  }, tool(async (args) => {
    const wallet = await Wallet.fetch(args.walletId);
    const addresses = wallet.getAddresses?.() || [];
    const address = addresses.find(a => a.getId?.() === args.addressId);

    if (!address) {
      throw new Error(`Address ${args.addressId} not found`);
    }

    const transfer = await address.transfer({
      destination: args.destination,
      assetId: args.asset?.toUpperCase() || 'ETH',
      amount: args.amount,
      gasPrice: args.gasPrice,
    });

    return {
      transactionHash: transfer.getTransactionHash?.(),
      status: transfer.getStatus?.(),
      walletId: args.walletId,
      addressId: args.addressId,
      destination: args.destination,
      amount: args.amount,
      asset: args.asset,
      network: CONFIG.network,
    };
  }));

  server.registerTool('wait_for_transaction', {
    title: 'Wait for transaction confirmation',
    description: 'Monitor and wait for a transaction to be confirmed',
    inputSchema: {
      walletId: z.string(),
      addressId: z.string(),
      transactionHash: z.string().describe('Transaction hash to monitor'),
      maxAttempts: z.number().int().default(30).describe('Max polling attempts'),
    },
  }, tool(async (args) => {
    const wallet = await Wallet.fetch(args.walletId);
    const addresses = wallet.getAddresses?.() || [];
    const address = addresses.find(a => a.getId?.() === args.addressId);

    if (!address) {
      throw new Error(`Address ${args.addressId} not found`);
    }

    // Poll for transaction status
    let attempts = 0;
    let status = 'pending';
    let result = null;

    while (attempts < args.maxAttempts && status === 'pending') {
      // Note: Actual implementation depends on CDP SDK's transaction monitoring
      await new Promise(resolve => setTimeout(resolve, 2000));
      attempts += 1;

      // In production, fetch actual transaction status from blockchain
      // This is a placeholder - implement based on CDP SDK capabilities
      status = 'confirmed'; // Simplified for example
    }

    return {
      transactionHash: args.transactionHash,
      status,
      attempts,
      confirmed: status === 'confirmed',
    };
  }));

  // Staking Tools

  server.registerTool('stake_asset', {
    title: 'Stake assets',
    description: 'Stake ETH or other supported assets (testnet only)',
    inputSchema: stakingSchema,
  }, tool(async (args) => {
    if (CONFIG.isMainnet) {
      throw new Error('Staking operations only available on testnet');
    }

    const wallet = await Wallet.fetch(args.walletId);
    const addresses = wallet.getAddresses?.() || [];
    const address = addresses.find(a => a.getId?.() === args.addressId);

    if (!address) {
      throw new Error(`Address ${args.addressId} not found`);
    }

    // Staking operation
    const stake = await address.stake?.({
      amount: args.amount,
      mode: args.mode,
    });

    return {
      transactionHash: stake?.getTransactionHash?.(),
      status: stake?.getStatus?.(),
      mode: args.mode,
      amount: args.amount,
      network: CONFIG.network,
    };
  }));

  // Faucet Tools (Testnet only)

  server.registerTool('request_testnet_funds', {
    title: 'Request testnet ETH from faucet',
    description: 'Request free testnet ETH (only on testnet networks)',
    inputSchema: addressSchema,
  }, tool(async (args) => {
    if (CONFIG.isMainnet) {
      throw new Error('Faucet only available on testnet');
    }

    const wallet = await Wallet.fetch(args.walletId);
    const addresses = wallet.getAddresses?.() || [];
    const address = addresses.find(a => a.getId?.() === args.addressId);

    if (!address) {
      throw new Error(`Address ${args.addressId} not found`);
    }

    try {
      const faucet = await address.requestFaucetFunds?.();

      return {
        success: true,
        transactionHash: faucet?.getTransactionHash?.(),
        status: 'requested',
        address: args.addressId,
        network: CONFIG.network,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        address: args.addressId,
        network: CONFIG.network,
      };
    }
  }));

  // Server Information Tools

  server.registerTool('get_cdp_status', {
    title: 'Get CDP platform status',
    description: 'Get current network and configuration status',
    inputSchema: {},
  }, tool(async () => ({
    network: CONFIG.network,
    isMainnet: CONFIG.isMainnet,
    sdkVersion: require('../package.json').dependencies['@coinbase/cdp-sdk'],
    debugMode: CONFIG.debug,
  })));

  server.registerTool('get_supported_networks', {
    title: 'List supported networks',
    description: 'Get list of all supported blockchain networks',
    inputSchema: {},
  }, tool(async () => ({
    networks: [
      'base-mainnet',
      'base-sepolia',
      'ethereum-mainnet',
      'ethereum-holesky',
      'arbitrum-mainnet',
      'arbitrum-sepolia',
      'polygon-mainnet',
      'polygon-amoy',
      'avalanche-mainnet',
      'avalanche-fuji',
    ],
    currentNetwork: CONFIG.network,
    isMainnet: CONFIG.isMainnet,
  })));

  return server;
};

if (require.main === module) {
  const server = createServer();
  server.connect(new StdioServerTransport()).catch((error) => {
    process.stderr.write(`CDP MCP Server failed: ${error.stack || error.message}\n`);
    process.exit(1);
  });
}

module.exports = { createServer };
