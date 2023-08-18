package de.motis_project.app2.io;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.preference.PreferenceManager;
import android.util.Log;

import java.io.IOException;

import de.motis_project.app2.journey.ConnectionWrapper;
import de.motis_project.app2.query.guesser.FavoritesDataSource;

public class Status {
    private static final boolean SERVER_SWITCH_AVAILABLE = false;
    private static final String SERVER_URL_RT = "wss://europe.motis-project.de/ws/";
    private static final String SERVER_URL_SOLL = SERVER_URL_RT;

    // Android Emulator
    // https://developer.android.com/studio/run/emulator-networking.html
    /*
    private static final String SERVER_URL_RT = "wss://10.0.2.2:8082";
    private static final String SERVER_URL_SOLL = "wss://10.0.2.2:8082";
    */

    private static final String TAG = "Status";
    private static final String SERVER_USE_RT = "server.use_rt";

    private static Status SINGLETON;
    private final MotisServer server;
    private FavoritesDataSource favoritesDb;
    private ConnectionWrapper connection;

    private Status(Context ctx, Handler handler) {
        favoritesDb = new FavoritesDataSource(ctx);

        final String serverUrl = getServerUrl(usingRtServer(ctx));

        Log.d(TAG, "init: serverUrl=" + serverUrl);

        server = new MotisServer(serverUrl, handler);
    }

    public static void init(Context ctx, Handler handler) {
        SINGLETON = new Status(ctx, handler);

        try {
            SINGLETON.getServer().connect();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getServerUrl(boolean useRt) {
        return useRt ? SERVER_URL_RT : SERVER_URL_SOLL;
    }

    public static synchronized Status get() {
        return SINGLETON;
    }

    public MotisServer getServer() {
        return server;
    }

    public ConnectionWrapper getConnection() {
        return connection;
    }

    public void setConnection(ConnectionWrapper connection) {
        this.connection = connection;
    }

    public FavoritesDataSource getFavoritesDb() {
        return favoritesDb;
    }

    public void switchServer(Context ctx, boolean useRt) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(ctx);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putBoolean(SERVER_USE_RT, useRt);
        editor.apply();

        server.setUrl(getServerUrl(useRt));
    }

    public boolean usingRtServer(Context ctx) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(ctx);

        return prefs.getBoolean(SERVER_USE_RT, false);
    }

    public boolean supportsServerSwitch() {
        return SERVER_SWITCH_AVAILABLE;
    }
}
