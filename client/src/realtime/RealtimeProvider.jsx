/**
 * Socket.IO bridge: pushes server events straight into the React Query cache,
 * so every component reading those queries updates without its own socket code.
 */
import { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { useQueryClient } from '@tanstack/react-query';
import { keys } from '../api/hooks';

const RealtimeContext = createContext({ connected: false });

export const RealtimeProvider = ({ children }) => {
  const queryClient = useQueryClient();
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const socket = io(import.meta.env.VITE_API_BASE || undefined, { transports: ['websocket', 'polling'] });
    socket.on('connect', () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));
    socket.on('ticker', (ticker) => queryClient.setQueryData(keys.ticker, ticker));
    socket.on('prediction', (prediction) => {
      queryClient.setQueryData(keys.prediction, prediction);
      queryClient.invalidateQueries({ queryKey: ['history'] });
      queryClient.invalidateQueries({ queryKey: ['accuracy'] });
    });
    socket.on('order', () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      queryClient.invalidateQueries({ queryKey: ['account'] });
    });
    return () => socket.disconnect();
  }, [queryClient]);

  return <RealtimeContext.Provider value={{ connected }}>{children}</RealtimeContext.Provider>;
};

export const useRealtime = () => useContext(RealtimeContext);
