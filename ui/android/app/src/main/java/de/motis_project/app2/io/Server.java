package de.motis_project.app2.io;

import android.os.Handler;
import android.util.Log;

import com.neovisionaries.ws.client.WebSocket;
import com.neovisionaries.ws.client.WebSocketAdapter;
import com.neovisionaries.ws.client.WebSocketException;
import com.neovisionaries.ws.client.WebSocketFactory;
import com.neovisionaries.ws.client.WebSocketFrame;
import com.neovisionaries.ws.client.WebSocketState;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import de.motis_project.app2.io.error.DisconnectedException;
import motis.Message;
import motis.MotisError;
import motis.MsgContent;

public class Server extends WebSocketAdapter {
    public interface Listener {
        void onMessage(Message m);

        void onConnect();

        void onDisconnect();
    }

    private static class SendQueueEntry {
        byte[] msg;
        Listener l;

        public SendQueueEntry(byte[] msg, Listener l) {
            this.msg = msg;
            this.l = l;
        }
    }

    private static final int RECONNECT_INTERVAL = 5000;
    private static final String TAG = "Server";

    private String url;
    private final List<Listener> listeners = new ArrayList<Listener>();
    private final Map<Listener, WebSocket> listenerSockets = new HashMap<>();
    private final Handler handler;
    private WebSocket ws;
    private long lastReceiveTimestamp = System.currentTimeMillis();
    private List<SendQueueEntry> sendQueue = new ArrayList<>();

    public Server(String url, Handler handler) {
        this.url = url;
        this.handler = handler;
        Log.i(TAG, "init url: " + url);
    }

    public void setUrl(String url) {
        synchronized (this) {
            if (!this.url.equals(url)) {
                Log.i(TAG, "URL changed to: " + url);
                this.url = url;
                Log.i(TAG, "Disconnecting...");
                disconnect();
                Log.i(TAG, "Connecting to new server...");
                tryConnect();
            }
        }
    }

    public void connect() throws IOException {
        synchronized (this) {
            if (ws != null) {
                WebSocketState state = ws.getState();
                Log.d(TAG, "existing websocket: uri=" + ws.getURI() + ", state=" +
                        state.toString() + ", id=" + ws.toString());
                if (state != WebSocketState.CLOSED && state != WebSocketState.CLOSING) {
                    Log.d(TAG, "websocket already connected");
                    if (!ws.getURI().toString().equals(this.url)) {
                        Log.i(TAG, "url changed, reconnecting");
                        disconnect();
                    } else {
                        return;
                    }
                }
            }

            WebSocketFactory factory = new WebSocketFactory();

            // Android Emulator
            /*
            try {
                factory.setSSLContext(NaiveSSLContext.getInstance("TLS"));
                factory.setVerifyHostname(false);
            } catch (NoSuchAlgorithmException e) {
                e.printStackTrace();
            }
            */

            factory.setConnectionTimeout(15000);

            Log.i(TAG, "connect url: " + url);
            ws = factory.createSocket(url);
            Log.d(TAG, "new websocket created: id=" + ws.toString());
            ws.addListener(this);
            ws.setPingInterval(60 * 1000);
            ws.connectAsynchronously();
        }
    }

    public void disconnect() {
        synchronized (this) {
            if (ws == null) {
                return;
            }
            ws.disconnect();
        }
    }

    public boolean isConnected() {
        if (ws == null) {
            System.out.println("NOT CONNECTED");
            return false;
        } else {
            System.out.println((ws.getState() == WebSocketState.OPEN) ? "CONNECTED" : "NOT CONNECTED");
            return ws.getState() == WebSocketState.OPEN;
        }
    }

    protected boolean send(byte[] msg, Listener l) throws DisconnectedException {
        WebSocket wsc = ws;
        if (wsc != null) {
            Log.d(TAG, "send: ws (uri=" + wsc.getURI() + ") state=" + wsc.getState());
        } else {
            Log.d(TAG, "send: ws == null");
        }
        if (!isConnected()) {
            Log.d(TAG, "adding message to send queue");
            synchronized (sendQueue) {
                sendQueue.add(new SendQueueEntry(msg, l));
            }
            scheduleConnect();
            return false;
        }
        Log.d(TAG, "sending message");
        synchronized (listeners) {
            listenerSockets.put(l, ws);
        }
        ws.sendPing();
        ws.sendBinary(msg);
        return true;
    }

    public void addListener(Listener l) {
        synchronized (listeners) {
            listeners.add(l);
        }
    }

    public void removeListener(Listener l) {
        synchronized (listeners) {
            listeners.remove(l);
            listenerSockets.remove(l);
        }
    }

    private void scheduleConnect() {
        System.out.println("Server.scheduleConnect");
        try {
            handler.postDelayed(() -> tryConnect(), RECONNECT_INTERVAL);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void tryConnect() {
        try {
            System.out.println("CONNECTING...");
            connect();
        } catch (IOException e) {
            System.out.println("CONNECT ERROR: " + e.getMessage());
            scheduleConnect();
        }
    }

    @Override
    public void onPongFrame(WebSocket websocket, WebSocketFrame frame) throws Exception {
        System.out.println("Server.onPongFrame");
    }

    @Override
    public void onBinaryMessage(WebSocket ws, byte[] buf) throws Exception {
        System.out.println("Server.onBinaryMessage");
        lastReceiveTimestamp = System.currentTimeMillis();

        Message msg = MessageBuilder.decode(buf);

        if (msg.contentType() == MsgContent.MotisError) {
            MotisError err = new MotisError();
            err = (MotisError) msg.content(err);
            System.out.println(
                    "RECEIVED ERROR: " + err.category() + ": " + err.reason());
        }

        ArrayList<Listener> currentListeners = new ArrayList<>(listeners);
        for (Listener l : currentListeners) {
            l.onMessage(msg);
        }
    }

    @Override
    public void onConnected(WebSocket websocket,
                            Map<String, List<String>> headers)
            throws Exception {
        System.out.println("Server.onConnected");

        lastReceiveTimestamp = System.currentTimeMillis();

        synchronized (sendQueue) {
            Log.d(TAG, "send queue contains " + sendQueue.size() + " messages");
            for (SendQueueEntry e : sendQueue) {
                send(e.msg, e.l);
            }
            sendQueue.clear();
        }

        synchronized (listeners) {
            for (Listener l : listeners) {
                if (listenerSockets.get(l) == ws) {
                    l.onConnect();
                }
            }
        }
    }

    @Override
    public void onDisconnected(WebSocket websocket,
                               WebSocketFrame serverCloseFrame,
                               WebSocketFrame clientCloseFrame,
                               boolean closedByServer) throws Exception {
        System.out.println("Server.onDisconnected");
        synchronized (listeners) {
            for (Listener l : listeners) {
                WebSocket ls = listenerSockets.get(l);
                if (ls == ws) {
                    l.onDisconnect();
                }
            }
        }
        scheduleConnect();
    }

    @Override
    public void onConnectError(WebSocket websocket,
                               WebSocketException exception) throws Exception {
        scheduleConnect();
    }
}
