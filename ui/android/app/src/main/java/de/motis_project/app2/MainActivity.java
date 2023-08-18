package de.motis_project.app2;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Switch;

import androidx.annotation.Nullable;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;

import com.google.android.material.navigation.NavigationView;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.query.QueryFragment;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PLAY_SERVICES_RESOLUTION_REQUEST = 9000;

    @BindView(R.id.drawer_layout)
    DrawerLayout drawerLayout;

    @BindView(R.id.content_frame)
    FrameLayout contentFrame;

    @BindView(R.id.main_toolbar)
    Toolbar mainToolbar;

    Switch fastSwitch;

    protected QueryFragment queryFragment;

    protected ServerChangeListener serverChangeListener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        Log.i(TAG, "onCreate");
        ButterKnife.bind(this);

        setSupportActionBar(mainToolbar);
        if (mainToolbar != null) {
            mainToolbar.setTitleTextColor(getResources().getColor(R.color.md_white));
        }
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null && drawerLayout != null) {
            ActionBarDrawerToggle drawerToggle = new ActionBarDrawerToggle(this, drawerLayout,
                mainToolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
            drawerLayout.addDrawerListener(drawerToggle);
            drawerToggle.syncState();
            drawerToggle.getDrawerArrowDrawable().setColor(getResources().getColor(R.color.md_white));
        }

        setSelectedNavItem(R.id.nav_search);
    }

    protected void setSelectedNavItem(int id) {
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
                setTitle(getResources().getString(R.string.search));
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
}
