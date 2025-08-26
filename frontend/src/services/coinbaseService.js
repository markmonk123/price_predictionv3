import { CdpClient } from '@coinbase/cdp-sdk';

const client = new CdpClient();

export async function fetchBitcoinPrice() {
  const ticker = await client.getProductTicker('BTC-USD');
  return ticker.price;
}
