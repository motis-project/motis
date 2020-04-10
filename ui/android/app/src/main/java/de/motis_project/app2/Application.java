package de.motis_project.app2;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.support.multidex.MultiDexApplication;

import org.acra.ACRA;
import org.acra.ReportingInteractionMode;
import org.acra.annotation.ReportsCrashes;

import de.motis_project.app2.io.Status;

@ReportsCrashes(
        formUri = "https://crash.motis-project.de",
        mode = ReportingInteractionMode.TOAST,
        resToastText = R.string.crash_toast_text,
        reportType = org.acra.sender.HttpSender.Type.JSON)
public class Application extends MultiDexApplication {
    @Override
    public void onCreate() {
        super.onCreate();
        Status.init(this, new Handler(Looper.getMainLooper()));
    }

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
        ACRA.init(this);
    }
}
