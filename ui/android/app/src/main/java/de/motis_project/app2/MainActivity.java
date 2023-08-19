package de.motis_project.app2;

import android.os.Bundle;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.query.QueryFragment;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    @BindView(R.id.drawer_layout)
    LinearLayout drawerLayout;

    @BindView(R.id.content_frame)
    FrameLayout contentFrame;

    protected QueryFragment queryFragment;

    protected ServerChangeListener serverChangeListener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        Log.i(TAG, "onCreate");
        ButterKnife.bind(this);

        if (queryFragment == null) {
            queryFragment = new QueryFragment();
        }
        setTitle(getResources().getString(R.string.search));
        FragmentManager fragmentManager = getSupportFragmentManager();
        fragmentManager.beginTransaction()
            .replace(R.id.content_frame, queryFragment)
            .commit();
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
