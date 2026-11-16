# Coinbase Developer Platform (CDP) MCP Server Setup Guide

## Overview
This guide shows how to set up the Coinbase Developer Platform (CDP) MCP server to enable programmatic wallet management, asset transfers, and staking operations through Claude Code and other MCP clients.

## Prerequisites
- Node.js 22.13+ ✅
- `@coinbase/cdp-sdk` 1.31.1+ ✅
- Coinbase account (free)
- CDP API credentials

## Step 1: Create CDP API Key

1. **Go to Coinbase Developer Platform**
   - Visit https://cdp.coinbase.com
   - Log in with your Coinbase account (create one if needed - it's free)

2. **Create API Key**
   - Navigate to **Settings → API Keys → Create Key**
   - Organization access: Choose appropriate level
   - Copy and **SAVE SECURELY**:
     - `API Key Name` (usually a UUID)
     - `Private Key` (PEM format - save to file)

3. **Choose Network**
   - Testnet (recommended for development): `base-sepolia`
   - Mainnet: `base-mainnet`

⚠️ **Security Warning**: Never commit private keys to git. Use environment variables only.

## Step 2: Configure Environment

Add your CDP credentials to `.env.development`:

```bash
# Coinbase Developer Platform (CDP)
CDP_API_KEY_NAME=your_key_name_here
CDP_API_KEY_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----
CDP_NETWORK=base-sepolia
CDP_DEBUG=false
```

### For Local Development
Create a `.env.local` file (git-ignored):
```bash
CDP_API_KEY_NAME=your_key_name
CDP_API_KEY_PRIVATE_KEY="$(cat /path/to/cdp_key.pem)"
```

### For Production
Use environment variables:
```bash
export CDP_API_KEY_NAME="your_key_name"
export CDP_API_KEY_PRIVATE_KEY="$(cat /path/to/cdp_key.pem)"
```

## Step 3: Verify Setup

Check that the CDP MCP server is registered:

```bash
cat .mcp.json | grep coinbase-cdp
```

Should show:
```json
"coinbase-cdp": {
  "command": "node",
  "args": ["mcp/cdp-server.js"]
}
```

## Step 4: Test Connection

### In VS Code / Claude Code
The CDP MCP server should now be available. Test with:

```
@mcp-server coinbase-cdp
tools: get_cdp_status
```

Should return:
```json
{
  "network": "base-sepolia",
  "isMainnet": false,
  "sdkVersion": "1.31.1+",
  "debugMode": false
}
```

### From Command Line
```bash
# Start the MCP server directly (for debugging)
CDP_API_KEY_NAME=your_key CDP_API_KEY_PRIVATE_KEY="$(cat key.pem)" \
  node mcp/cdp-server.js

# Should connect and wait for stdio commands (Ctrl+C to exit)
```

## Available Tools

### Wallet Management
- `create_wallet` - Create a new wallet
- `list_wallets` - List all wallets
- `get_wallet` - Get wallet details

### Address Management  
- `create_address` - Create new address in wallet
- `list_addresses` - List wallet addresses

### Assets & Balances
- `get_balance` - Get address balances

### Transfers
- `transfer_asset` - Transfer ETH, USDC, or other assets
- `wait_for_transaction` - Monitor transaction status

### Staking (Testnet)
- `stake_asset` - Stake assets
- `request_testnet_funds` - Request free testnet ETH from faucet

### Platform
- `get_cdp_status` - Check network and config
- `get_supported_networks` - List available networks

## Example Workflow

### 1. Create a Wallet
```
Tool: create_wallet
Args: { "name": "My Trading Wallet" }

Returns: {
  "id": "wallet-uuid",
  "network": "base-sepolia",
  "addresses": [...],
  "displayName": "My Trading Wallet"
}
```

### 2. Get Testnet Funds
```
Tool: request_testnet_funds
Args: {
  "walletId": "wallet-uuid",
  "addressId": "address-uuid"
}

Returns: {
  "success": true,
  "transactionHash": "0x...",
  "status": "requested"
}
```

### 3. Check Balance
```
Tool: get_balance
Args: {
  "walletId": "wallet-uuid",
  "addressId": "address-uuid"
}

Returns: {
  "balances": [
    { "token": "ETH", "amount": "1.5" },
    { "token": "USDC", "amount": "100" }
  ]
}
```

### 4. Transfer Assets
```
Tool: transfer_asset
Args: {
  "walletId": "wallet-uuid",
  "addressId": "address-uuid",
  "destination": "0x...",
  "asset": "eth",
  "amount": "0.1"
}

Returns: {
  "transactionHash": "0x...",
  "status": "pending"
}
```

## Troubleshooting

### "CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY required"
**Fix**: Add credentials to `.env.development` or set environment variables

### "Network not found: base-sepolia"
**Fix**: Check `CDP_NETWORK` value, must be one of:
- `base-sepolia` (testnet, default)
- `base-mainnet`
- `ethereum-holesky`, `ethereum-mainnet`
- `arbitrum-sepolia`, `arbitrum-mainnet`
- etc.

### "Wallet not found"
**Fix**: Ensure wallet ID is correct and was created on the same network

### "MCP server won't connect"
**Steps**:
1. Verify `.mcp.json` has `coinbase-cdp` entry
2. Check node_modules has `@coinbase/cdp-sdk`
3. Test with `node mcp/cdp-server.js` directly
4. Check for errors in stderr output
5. Enable `CDP_DEBUG=true` for verbose logging

## Network Selection

### Testnet (Development)
- **Network**: `base-sepolia`, `ethereum-holesky`, etc.
- **Faucet**: Free testnet ETH available
- **Purpose**: Testing, learning, development
- **Note**: No real value, funds lost after period

### Mainnet (Production)
- **Network**: `base-mainnet`, `ethereum-mainnet`, etc.
- **Faucet**: Not available
- **Purpose**: Real transactions, live trading
- **Warning**: Real funds! Use with caution

## Integration with Trading Platform

The CDP MCP server works alongside your existing price prediction platform:

1. **Price Prediction Service** (`mcp/server.js`)
   - Provides: `run_prediction`, market data, order management
   - Uses: Local paper broker + Coinbase Advanced Trade API

2. **CDP Wallet Service** (`mcp/cdp-server.js`)
   - Provides: Wallet management, asset transfers, staking
   - Uses: Coinbase Programmable Wallet API

Together they enable:
- Automated trading with predictive models
- Multi-asset portfolio management
- Testnet experimentation before mainnet trading
- Staking operations integrated with trading logic

## Security Best Practices

1. ✅ Never commit private keys to git
2. ✅ Use `.env.local` for sensitive local config
3. ✅ Rotate API keys regularly
4. ✅ Use testnet for development/testing
5. ✅ Set `CDP_DEBUG=false` in production
6. ✅ Restrict API key permissions in Coinbase settings
7. ✅ Use environment variables in CI/CD, never hardcoded
8. ✅ Monitor wallet transactions regularly

## Next Steps

1. ✅ Create Coinbase account and CDP credentials
2. ✅ Add credentials to `.env.development`
3. ✅ Verify MCP server connection
4. ✅ Test wallet operations
5. ✅ Integrate with your trading logic
6. ✅ Move to testnet for realistic testing
7. ✅ Plan mainnet migration carefully

## Resources

- [Coinbase Developer Platform](https://cdp.coinbase.com)
- [CDP API Documentation](https://docs.cdp.coinbase.com)
- [CDP SDK (Node.js)](https://github.com/coinbase/coinbase-sdk-js)
- [Base Network](https://www.base.org)
- [Model Context Protocol](https://modelcontextprotocol.io)

## Support

For issues:
1. Check CDP API status at https://status.coinbase.com
2. Enable debug mode: `CDP_DEBUG=true`
3. Verify credentials are correct
4. Check `.mcp.json` configuration
5. Review server logs in stderr
