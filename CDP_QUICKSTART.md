## Coinbase Developer Platform (CDP) MCP - Quick Start

### 1️⃣ Get CDP Credentials (2 minutes)

**Visit Coinbase Developer Platform:**
```
https://cdp.coinbase.com → Login → Settings → API Keys → Create Key
```

**Save these securely:**
- `API Key Name` (looks like a UUID)
- `Private Key` (long text starting with `-----BEGIN EC PRIVATE KEY-----`)

### 2️⃣ Configure Environment

**Edit `.env.development`:**
```bash
# Add these three lines:
CDP_API_KEY_NAME=your_key_name_here
CDP_API_KEY_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----
...
-----END EC PRIVATE KEY-----
CDP_NETWORK=base-sepolia
```

**Or use a separate `.env.local` file:**
```bash
# Create file and add credentials (this file is git-ignored)
echo 'CDP_API_KEY_NAME=...' > .env.local
```

### 3️⃣ Start Development Environment

```bash
npm run dev-full
```

Both servers will start:
- Backend: http://localhost:5000 (price prediction MCP server)
- Frontend: http://localhost:5173 (React UI)
- CDP MCP: Ready for VS Code / Claude Code

### 4️⃣ Use from VS Code / Claude Code

#### Create a Wallet
```
@mcp-server coinbase-cdp
I want to create a wallet for testnet trading.
```

#### Get Testnet Funds
```
@mcp-server coinbase-cdp
I just created wallet with ID [wallet-id].
Get testnet ETH from the faucet for address [address-id].
```

#### Check Balances
```
@mcp-server coinbase-cdp
Show me the balances for wallet [wallet-id] address [address-id].
```

#### Transfer Assets
```
@mcp-server coinbase-cdp
Transfer 0.1 ETH from wallet [wallet-id] to address 0x123... 
```

### 5️⃣ Network Choices

**For Learning/Testing (Recommended):**
- Network: `base-sepolia`
- Free testnet ETH via faucet
- No real money at risk

**For Live Trading:**
- Network: `base-mainnet`
- Real funds required
- Permanent transactions

### 6️⃣ Available Commands

| Command | Purpose |
|---------|---------|
| `create_wallet` | Create new wallet |
| `list_wallets` | See all wallets |
| `create_address` | New address in wallet |
| `get_balance` | Check ETH/USDC balance |
| `transfer_asset` | Send ETH/USDC |
| `request_testnet_funds` | Get free testnet ETH |
| `stake_asset` | Stake ETH (testnet only) |
| `get_cdp_status` | Check connection status |

### Troubleshooting

**"MCP server won't connect"**
- Check credentials in `.env.development`
- Verify `.mcp.json` has `coinbase-cdp` entry
- Run: `node mcp/cdp-server.js` to see errors

**"Invalid API key"**
- Generate new key at https://cdp.coinbase.com
- Copy key name and private key exactly
- No extra spaces or newlines

**"Network not found"**
- Use testnet: `base-sepolia`
- Or mainnet: `base-mainnet`
- Check `.env.development` CDP_NETWORK setting

### Integration Example

Combine price prediction with wallet management:

```
1. Run prediction model → Get BTC price direction forecast
2. If prediction is bullish:
   - Create/use wallet
   - Buy ETH/BTC via Coinbase API
   - Monitor position via MCP tools
3. If prediction is bearish:
   - Transfer assets to cold storage
   - Update portfolio balance
```

### Next: Live Trading Setup

Once comfortable with testnet:
1. Switch to mainnet: `CDP_NETWORK=base-mainnet`
2. Add real funds to wallet
3. Implement risk management rules
4. Start automated trading

### Support

- **CDP Docs**: https://docs.cdp.coinbase.com
- **SDK Reference**: https://github.com/coinbase/coinbase-sdk-js
- **This Project**: See `CDP_SETUP.md` for detailed guide

---

**Status**: ✅ MCP server ready
**Network**: Testnet (base-sepolia)
**Next**: Add your CDP credentials and start experimenting!
