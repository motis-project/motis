package de.motis_project.app2;

import android.os.Handler;
import android.os.Looper;
import androidx.multidex.MultiDexApplication;

import de.motis_project.app2.io.Status;

public class Application extends MultiDexApplication {
    @Override
    public void onCreate() {
        super.onCreate();
        Status.init(this, new Handler(Looper.getMainLooper()));
    }
}
