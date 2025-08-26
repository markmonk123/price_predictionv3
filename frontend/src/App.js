import React, { useEffect, useState } from 'react';
import { fetchBitcoinPrice } from './services/coinbaseService';

function App() {
  const [price, setPrice] = useState(null);

  useEffect(() => {
    fetchBitcoinPrice()
      .then(setPrice)
      .catch((err) => console.error('Failed to fetch price', err));
  }, []);

  return (
    <div>
      <h1>Bitcoin Price</h1>
      {price ? <p>${price}</p> : <p>Loading...</p>}
    </div>
  );
}

export default App;
