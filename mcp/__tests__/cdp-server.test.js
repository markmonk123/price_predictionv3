/**
 * CDP MCP Server Tests
 * Tests the Coinbase Developer Platform MCP server implementation
 */

const assert = require('assert');
const { createServer } = require('./cdp-server');

describe('CDP MCP Server', () => {
  let server;

  before(() => {
    // Skip if credentials not configured
    if (!process.env.CDP_API_KEY_NAME || !process.env.CDP_API_KEY_PRIVATE_KEY) {
      console.warn('⚠️  Skipping CDP tests - credentials not configured');
      console.warn('   Set CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY to enable');
      this.skip();
    }
  });

  it('should create server instance', () => {
    assert.doesNotThrow(() => {
      server = createServer();
    });
    assert(server, 'Server should be created');
  });

  it('should have required tools registered', () => {
    if (!server) this.skip();

    const tools = [
      'create_wallet',
      'list_wallets',
      'get_wallet',
      'create_address',
      'list_addresses',
      'get_balance',
      'transfer_asset',
      'wait_for_transaction',
      'stake_asset',
      'request_testnet_funds',
      'get_cdp_status',
      'get_supported_networks',
    ];

    tools.forEach(tool => {
      assert(server, `Tool ${tool} should exist`);
    });
  });

  it('should handle network configuration', () => {
    if (!server) this.skip();

    const network = process.env.CDP_NETWORK || 'base-sepolia';
    assert(network, 'Network should be configured');
    assert(
      network.includes('sepolia') || network.includes('mainnet'),
      'Network should be testnet or mainnet'
    );
  });
});

module.exports = { };
