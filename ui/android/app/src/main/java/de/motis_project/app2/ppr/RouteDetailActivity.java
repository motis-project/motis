package de.motis_project.app2.ppr;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v4.view.ViewCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;

import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.LatLngBounds;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.PolylineOptions;

import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.R;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.lib.AnchoredBottomSheetBehavior;
import de.motis_project.app2.lib.MapBehavior;
import de.motis_project.app2.ppr.route.RouteWrapper;
import de.motis_project.app2.ppr.route.StepInfo;

public class RouteDetailActivity
        extends AppCompatActivity
        implements OnMapAndViewReadyListener.OnGlobalLayoutAndMapReadyListener,
        RouteStepsAdapter.OnClickListener {
    private static final String TAG = "RouteDetailActivity";
    public static final int MAP_ZOOM_PADDING = 50;
    private RouteWrapper route;
    private RouteStepsAdapter stepsAdapter;

    private GoogleMap map;
    private SupportMapFragment mapFragment;
    private View mapView;

    @BindView(R.id.route_steps)
    RouteStepsView stepsView;

    @BindView(R.id.toolbar)
    Toolbar toolbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ppr_route_detail_activity);
        ButterKnife.bind(this);

        route = Status.get().getPprRoute();
        stepsAdapter = new RouteStepsAdapter(route);
        stepsAdapter.setClickListener(this);
        stepsView.setAdapter(stepsAdapter);

        // Obtain the SupportMapFragment and get notified when the map is ready to be used.
        mapFragment = (SupportMapFragment) getSupportFragmentManager().findFragmentById(R.id.map);
        mapView = mapFragment.getView();
        new OnMapAndViewReadyListener(mapFragment, this);

        final View view = this.findViewById(android.R.id.content);
        final AnchoredBottomSheetBehavior bsb = AnchoredBottomSheetBehavior.from(stepsView);
        float density = getResources().getDisplayMetrics().density;
        bsb.setDragZoneHeight(60 * density);
        view.post(() -> {
            int anchorHeight = (int) (view.getHeight() * .6);
            bsb.setAnchorHeight(anchorHeight);
        });

        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        ViewCompat.setElevation(toolbar, 4 * density);
        ViewCompat.setElevation(stepsView, 8 * density);

        MapBehavior mapBehavior = MapBehavior.from(mapView);
        mapBehavior.setBottomView(stepsView);
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        map = googleMap;
        // https://developers.google.com/android/reference/com/google/android/gms/maps/GoogleMap#setPadding(int,%20int,%20int,%20int)
        map.setPadding(0, toolbar.getHeight(), 0, 0); // left, top, right, bottom

        LatLng start = route.getStart();
        LatLng destination = route.getDestination();

        map.addMarker(new MarkerOptions()
                .position(start)
                .title(getResources().getString(R.string.start)));
        map.addMarker(new MarkerOptions()
                .position(destination)
                .title(getResources().getString(R.string.destination))
                .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE)));

        Iterable<LatLng> routePath = route.getPath();
        map.addPolyline(new PolylineOptions()
                .addAll(routePath)
                .color(Color.BLUE));

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED) {
            map.setMyLocationEnabled(true);
        }

        final LatLngBounds bounds = MapUtil.getBounds(routePath);
        final int padding = (int) (MAP_ZOOM_PADDING * getResources().getDisplayMetrics().density);
        try {
            map.moveCamera(CameraUpdateFactory.newLatLngBounds(bounds, padding));
        } catch (IllegalStateException ex) {
            mapView.post(() -> {
                try {
                    map.moveCamera(CameraUpdateFactory.newLatLngBounds(bounds, padding));
                } catch (IllegalStateException ex2) {
                    ex2.printStackTrace();
                }
            });
        }
    }

    @Override
    public void onStepClicked(StepInfo step) {
        LatLngBounds pathBounds = MapUtil.getBounds(step.getPath());
        setBottomSheetAnchored();
        final int padding = (int) (MAP_ZOOM_PADDING * getResources().getDisplayMetrics().density);
        try {
            map.animateCamera(CameraUpdateFactory.newLatLngBounds(pathBounds, padding));
        } catch (IllegalStateException ex) {
            ex.printStackTrace();
        }
    }

    private void setBottomSheetAnchored() {
        AnchoredBottomSheetBehavior bsb = AnchoredBottomSheetBehavior.from(stepsView);
        bsb.setState(AnchoredBottomSheetBehavior.STATE_ANCHORED);
//        stepsView.getLayoutManager().scrollToPosition(0);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        } else {
            return super.onOptionsItemSelected(item);
        }
    }
}
