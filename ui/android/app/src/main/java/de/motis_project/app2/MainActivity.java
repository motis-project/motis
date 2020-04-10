package de.motis_project.app2;

import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.design.widget.NavigationView;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.support.v4.app.FragmentManager;
import android.support.v4.widget.DrawerLayout;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Switch;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GoogleApiAvailability;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.intermodal.IntermodalFragment;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.ppr.PPRFragment;
import de.motis_project.app2.query.QueryFragment;
import de.motis_project.app2.saved.SavedConnectionsFragment;

public class MainActivity extends FragmentActivity {
    private static final String TAG = "MainActivity";
    private static final int PLAY_SERVICES_RESOLUTION_REQUEST = 9000;

    @BindView(R.id.drawer_layout)
    DrawerLayout drawerLayout;

    @BindView(R.id.content_frame)
    FrameLayout contentFrame;

    @BindView(R.id.navigation)
    NavigationView navigationView;

    Switch fastSwitch;

    protected QueryFragment queryFragment;
    protected SavedConnectionsFragment savedConnectionsFragment;
    protected PPRFragment pprFragment;
    protected IntermodalFragment intermodalFragment;

    protected ServerChangeListener serverChangeListener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        Log.i(TAG, "onCreate");
        ButterKnife.bind(this);

        fastSwitch = (Switch) ((RelativeLayout) navigationView.getMenu()
                .findItem(R.id.nav_fast_switch).getActionView())
                .getChildAt(0);
        fastSwitch.setChecked(!Status.get().usingRtServer(this));
        fastSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            boolean useRt = !isChecked;
            Log.i(TAG, "Switching server: rt=" + useRt);
            Status.get().switchServer(this, useRt);
            Log.d(TAG, "Notifying server change listener");
            ServerChangeListener scl = serverChangeListener;
            if (scl != null) {
                scl.serverChanged();
            }
        });
        if (!Status.get().supportsServerSwitch()) {
            navigationView.getMenu().setGroupVisible(R.id.nav_server_grp, false);
        }

        navigationView.setNavigationItemSelectedListener(item -> {
            item.setChecked(true);
            boolean selected = showTab(item.getItemId());
            if (selected) {
                drawerLayout.closeDrawers();
            }
            return selected;
        });

        setSelectedNavItem(R.id.nav_search);
    }

    protected void setSelectedNavItem(int id) {
        navigationView.setCheckedItem(id);
        showTab(id);
    }

    protected boolean showTab(int itemId) {
        Fragment selectedFragment = null;
        switch (itemId) {
            case R.id.nav_search:
                if (queryFragment == null) {
                    queryFragment = new QueryFragment();
                }
                selectedFragment = queryFragment;
                break;
            /*
            case R.id.nav_connections:
                if (savedConnectionsFragment == null) {
                    savedConnectionsFragment = new SavedConnectionsFragment();
                }
                selectedFragment = savedConnectionsFragment;
                break;
            */
            case R.id.nav_ppr:
                if (pprFragment == null) {
                    pprFragment = new PPRFragment();
                }
                selectedFragment = pprFragment;
                break;
            case R.id.nav_intermodal:
                if (intermodalFragment == null) {
                    intermodalFragment = new IntermodalFragment();
                }
                selectedFragment = intermodalFragment;
                break;
            case R.id.nav_fast_switch:
                fastSwitch.toggle();
                return false;
            default:
                Log.e(TAG, "Unknown navigation item selected: " + itemId);
                return false;
        }
        FragmentManager fragmentManager = getSupportFragmentManager();
        fragmentManager.beginTransaction()
                .replace(R.id.content_frame, selectedFragment)
                .commit();
        return true;
    }

    @Override
    public void onAttachFragment(Fragment fragment) {
        Log.i(TAG, "Fragment attached: " + fragment.getClass().getName());
        if (fragment instanceof ServerChangeListener) {
            serverChangeListener = (ServerChangeListener) fragment;
        } else {
            serverChangeListener = null;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        boolean playServices = checkGooglePlayServices();
        Log.i(TAG, "Goole Play Services available: " + playServices);
    }

    protected boolean checkGooglePlayServices() {
        GoogleApiAvailability api = GoogleApiAvailability.getInstance();
        int code = api.isGooglePlayServicesAvailable(this);
        if (code == ConnectionResult.SUCCESS) {
            return true;
        }
        Log.w(TAG, "Google Play Services not available: " + code);
        if (api.isUserResolvableError(code)) {
            Log.w(TAG, "Google Play Services error is user resolvable, showing dialog");
            api.getErrorDialog(this, code, PLAY_SERVICES_RESOLUTION_REQUEST).show();
        } else {
            Log.e(TAG, "Device not supported by Google Play Services");
        }
        return false;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        switch (requestCode) {
            case PLAY_SERVICES_RESOLUTION_REQUEST:
                Log.i(TAG, "Google Play Services resolution request returned: " + resultCode);
                boolean playServices = checkGooglePlayServices();
                Log.i(TAG, "Google Play Services available: " + playServices);
                break;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
