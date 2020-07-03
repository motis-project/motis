package de.motis_project.app2.detail;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.support.v4.view.ViewCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.LatLngBounds;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;
import com.google.android.gms.maps.model.PolylineOptions;

import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Date;
import java.util.HashSet;
import java.util.List;

import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.journey.ConnectionWrapper;
import de.motis_project.app2.journey.CopyConnection;
import de.motis_project.app2.lib.AnchoredBottomSheetBehavior;
import de.motis_project.app2.lib.MapBehavior;
import de.motis_project.app2.ppr.MapUtil;
import de.motis_project.app2.ppr.OnMapAndViewReadyListener;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import de.motis_project.app2.ppr.route.RouteWrapper;
import de.motis_project.app2.ppr.route.StepInfo;
import motis.Connection;
import motis.Move;
import motis.MoveWrapper;
import motis.Stop;
import motis.Transport;

public class IntermodalDetailActivity extends AppCompatActivity
        implements DetailClickHandler,
        OnMapAndViewReadyListener.OnGlobalLayoutAndMapReadyListener {
    private static final String TAG = "IntermodalDetailActivit";
    public static final String SHOW_SAVE_ACTION = "SHOW_SAVE_ACTION";
    public static final int MAP_ZOOM_PADDING = 50;

    private ConnectionWrapper con;
    private PprSearchOptions pprSearchOptions;
    private HashSet<JourneyUtil.Section> expandedSections = new HashSet<>();

    @BindString(R.string.transfer)
    String transfer;
    @BindString(R.string.transfers)
    String transfers;
    @BindString(R.string.connection_saved)
    String connectionSaved;

    @BindView(R.id.detail_dep_station)
    TextView depStation;
    @BindView(R.id.detail_arr_station)
    TextView arrStation;
    @BindView(R.id.detail_dep_schedule_time)
    TextView depSchedTime;
    @BindView(R.id.detail_arr_schedule_time)
    TextView arrSchedTime;
    @BindView(R.id.detail_travel_duration)
    TextView travelDuration;
    @BindView(R.id.detail_number_of_transfers)
    TextView numberOfTransfers;
    @BindView(R.id.detail_journey_details)
    LinearLayout journeyDetails;
    @BindView(R.id.bottom_sheet)
    LinearLayout bottomSheet;
    @BindView(R.id.detail_header_inner)
    LinearLayout detailHeader;
    @BindView(R.id.toolbar)
    Toolbar toolbar;

    private GoogleMap map;
    private SupportMapFragment mapFragment;
    private View mapView;
    private AnchoredBottomSheetBehavior bsb;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.detail_intermodal);
        ButterKnife.bind(this);

        con = Status.get().getConnection();
        pprSearchOptions = Status.get().getPprSearchOptions();

        String formattedDate = SimpleDateFormat
                .getDateInstance(java.text.DateFormat.SHORT)
                .format(new Date(con.getFirstStop().departure().scheduleTime() * 1000));
        setTitle(formattedDate);

        // Obtain the SupportMapFragment and get notified when the map is ready to be used.
        mapFragment = (SupportMapFragment) getSupportFragmentManager().findFragmentById(R.id.map);
        mapView = mapFragment.getView();

        final View view = this.findViewById(android.R.id.content);
        bsb = AnchoredBottomSheetBehavior.from(bottomSheet);
        float density = getResources().getDisplayMetrics().density;
        view.post(() -> {
            int anchorHeight = (int) (view.getHeight() * .6);
            bsb.setAnchorHeight(anchorHeight);
            bsb.setPeekHeight(detailHeader.getHeight());
            bsb.setDragZoneHeight(detailHeader.getHeight());
        });

        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        ViewCompat.setElevation(toolbar, 4 * density);
        ViewCompat.setElevation(bottomSheet, 8 * density);

        MapBehavior mapBehavior = MapBehavior.from(mapView);
        mapBehavior.setBottomView(bottomSheet);

        new OnMapAndViewReadyListener(mapFragment, this);

        initHeader();
        create();
    }

    void initHeader() {
        depStation.setText(con.getStartName());
        arrStation.setText(con.getDestinationName());

        long depTime = con.getFirstStop().departure().scheduleTime();
        long arrTime = con.getLastStop().arrival().scheduleTime();
        depSchedTime.setText(TimeUtil.formatTime(depTime));
        arrSchedTime.setText(TimeUtil.formatTime(arrTime));

        long minutes = (arrTime - depTime) / 60;
        travelDuration.setText(TimeUtil.formatDuration(minutes));

        int transferCount = con.getNumberOfTransfers();
        String transferPlural = (transferCount == 1) ? transfer : transfers;
        numberOfTransfers.setText(String.format(transferPlural, transferCount));
    }

    void create() {
        TransportBuilder.setConnection(getLayoutInflater(),
                journeyDetails, con, pprSearchOptions, expandedSections);
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        create();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.journey_detail_toolbar, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public void expandSection(JourneyUtil.Section section) {
        expandedSections.add(section);
        create();
    }

    @Override
    public void contractSection(JourneyUtil.Section section) {
        expandedSections.remove(section);
        create();
    }

    @Override
    public void refreshSection(JourneyUtil.Section section) {
        create();
    }

    @Override
    public void transportStopClicked(Stop stop) {
        LatLngBounds pathBounds = MapUtil.toBounds(MapUtil.toLatLng(stop.station().pos()), 100);
        setBottomSheetAnchored();
        moveMapCamera(pathBounds);
    }

    @Override
    public void walkStepClicked(StepInfo stepInfo) {
        LatLngBounds pathBounds = MapUtil.getBounds(stepInfo.getPath());
        setBottomSheetAnchored();
        moveMapCamera(pathBounds);
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        map = googleMap;
        // https://developers.google.com/android/reference/com/google/android/gms/maps/GoogleMap#setPadding(int,%20int,%20int,%20int)
        map.setPadding(0, toolbar.getHeight(), 0, 0); // left, top, right, bottom

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED) {
            map.setMyLocationEnabled(true);
        }

        addConnectionToMap();
    }

    protected void addConnectionToMap() {
        map.clear();
        Connection c = con.getConnection();
        LatLngBounds.Builder boundsBuilder = LatLngBounds.builder();
        LatLng start = con.getStartLatLng();
        LatLng destination = con.getDestinationLatLng();
        boundsBuilder.include(start).include(destination);

        map.addMarker(new MarkerOptions()
                .position(start)
                .title(getResources().getString(R.string.start)));
        map.addMarker(new MarkerOptions()
                .position(destination)
                .title(getResources().getString(R.string.destination))
                .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE)));


        RouteWrapper leadingWalk = con.getLeadingWalkRoute();
        RouteWrapper trailingWalk = con.getTrailingWalkRoute();

        if (leadingWalk != null) {
            addWalkPolyline(leadingWalk, boundsBuilder);
        }
        if (trailingWalk != null) {
            addWalkPolyline(trailingWalk, boundsBuilder);
        }

        List<JourneyUtil.Section> sections = JourneyUtil.getSections(c, false);
        JourneyUtil.Section prevSection = null;
        for (JourneyUtil.Section section : sections) {
            MoveWrapper m = JourneyUtil.getMove(c, section);
            if (prevSection != null && prevSection.to != section.from) {
                addTransportPolyline(c.stops(prevSection.to), c.stops(section.from), JourneyUtil.WALK_CLASS, boundsBuilder);
            }
            if (m.moveType() == Move.Transport) {
                Transport t = JourneyUtil.getTransport(m);
                addTransportPolyline(c.stops(section.from), c.stops(section.to), t.clasz(), boundsBuilder);
            } else if (m.moveType() == Move.Walk) {
//                Walk w = JourneyUtil.getWalk(m);
                addTransportPolyline(c.stops(section.from), c.stops(section.to), JourneyUtil.WALK_CLASS, boundsBuilder);
            }
            prevSection = section;
        }

        final LatLngBounds bounds = boundsBuilder.build();
        moveMapCamera(bounds);
    }

    protected void moveMapCamera(LatLngBounds bounds) {
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

    protected Polyline addWalkPolyline(RouteWrapper walkRoute,
                                       @Nullable LatLngBounds.Builder boundsBuilder) {
        Collection<LatLng> path = walkRoute.getPath();
        if (boundsBuilder != null) {
            for (LatLng pt : path) {
                boundsBuilder.include(pt);
            }
        }
        return map.addPolyline(new PolylineOptions()
                .addAll(path)
                .color(JourneyUtil.getColor(this, JourneyUtil.WALK_CLASS))
                .width(20));
    }

    protected Polyline addTransportPolyline(Stop from, Stop to, long clasz,
                                            @Nullable LatLngBounds.Builder boundsBuilder) {
        LatLng fromLatLng = MapUtil.toLatLng(from.station().pos());
        LatLng toLatLng = MapUtil.toLatLng(to.station().pos());
        if (boundsBuilder != null) {
            boundsBuilder.include(fromLatLng).include(toLatLng);
        }
        return map.addPolyline(new PolylineOptions()
                .add(fromLatLng, toLatLng)
                .color(JourneyUtil.getColor(this, clasz))
                .width(20));
    }

    @OnClick(R.id.detail_header_inner)
    protected void onDetailHeaderClick() {
        switch (bsb.getState()) {
            case AnchoredBottomSheetBehavior.STATE_COLLAPSED:
                bsb.setState(AnchoredBottomSheetBehavior.STATE_ANCHORED);
                break;
            case AnchoredBottomSheetBehavior.STATE_ANCHORED:
                bsb.setState(AnchoredBottomSheetBehavior.STATE_COLLAPSED);
                break;
            case AnchoredBottomSheetBehavior.STATE_EXPANDED:
                bsb.setState(AnchoredBottomSheetBehavior.STATE_ANCHORED);
                break;
            default:
                break;
        }
    }

    private void setBottomSheetAnchored() {
        bsb.setState(AnchoredBottomSheetBehavior.STATE_ANCHORED);
    }
}
