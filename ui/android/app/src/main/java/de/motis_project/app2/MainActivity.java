package de.motis_project.app2;

import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

import androidx.appcompat.app.AlertDialog;
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

        showPrivacyDialog();
    }

    void showPrivacyDialog() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        if (!prefs.getBoolean("show_privacy", true)) {
            return;
        }

        AlertDialog.Builder alertDialog = new AlertDialog.Builder(this);
        alertDialog.setTitle(R.string.data_protection_headline);
        alertDialog.setMessage(R.string.data_protection);
        alertDialog.setPositiveButton(R.string.read,
            new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    startActivity(new Intent(
                        Intent.ACTION_VIEW, Uri
                        .parse("https://motis-project.de/privacy")));
                    prefs.edit().putBoolean("show_privacy", false).commit();
                    dialog.dismiss();
                }
            });

        alertDialog.setNeutralButton(R.string.dont_show_again,
            new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    prefs.edit().putBoolean("show_privacy", false).commit();
                    dialog.dismiss();
                }
            });

        alertDialog.setNegativeButton(R.string.show_again_later,
            new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    dialog.dismiss();
                }
            });

        alertDialog.show();
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
