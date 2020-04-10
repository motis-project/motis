package de.motis_project.app2.intermodal;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.design.widget.AppBarLayout;
import android.support.v4.app.DialogFragment;
import android.support.v4.app.Fragment;
import android.support.v4.content.ContextCompat;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import com.google.android.gms.common.GooglePlayServicesNotAvailableException;
import com.google.android.gms.common.GooglePlayServicesRepairableException;
import com.google.android.libraries.places.compat.Place;
import com.google.android.libraries.places.compat.ui.PlacePicker;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.LatLngBounds;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.Polyline;
import com.google.android.gms.maps.model.PolylineOptions;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import butterknife.OnItemSelected;
import de.motis_project.app2.R;
import de.motis_project.app2.ServerChangeListener;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.detail.IntermodalDetailActivity;
import de.motis_project.app2.intermodal.journey.WalkCache;
import de.motis_project.app2.intermodal.journey.WalkKey;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.io.error.MotisErrorException;
import de.motis_project.app2.journey.BaseJourneyListView;
import de.motis_project.app2.journey.ConnectionWrapper;
import de.motis_project.app2.journey.ServerErrorView;
import de.motis_project.app2.lib.AnchoredBottomSheetBehavior;
import de.motis_project.app2.lib.MapBehavior;
import de.motis_project.app2.ppr.MapUtil;
import de.motis_project.app2.ppr.NamedLocation;
import de.motis_project.app2.ppr.OnMapAndViewReadyListener;
import de.motis_project.app2.ppr.profiles.SearchProfile;
import de.motis_project.app2.ppr.profiles.SearchProfiles;
import de.motis_project.app2.ppr.route.RouteWrapper;
import de.motis_project.app2.query.DatePickerDialogFragment;
import de.motis_project.app2.query.TimePickerDialogFragment;
import motis.routing.RoutingResponse;

import static android.app.Activity.RESULT_OK;

public class IntermodalFragment extends Fragment
        implements android.app.DatePickerDialog.OnDateSetListener,
        TimePickerDialogFragment.ChangeListener,
        GoogleMap.OnPolylineClickListener,
        BaseJourneyListView.LoadResultListener,
        WalkCache.Listener, ServerChangeListener {

    private static final String TAG = "IntermodalFragment";

    private static final String PREFS_NAME = "intermodal";
    private static final String PREFS_QUERY_PREFIX = "query.";

    private static final int FROM_PLACE_PICKER_REQUEST = 1;
    private static final int TO_PLACE_PICKER_REQUEST = 2;
    private static final int FROM_LOCATION_PERMISSION_REQUEST = 1;
    private static final int TO_LOCATION_PERMISSION_REQUEST = 2;

    private Context context;

    private IntermodalQuery query;
    private IntermodalConnectionLoader connectionLoader;
    private final SearchProfiles searchProfiles = new SearchProfiles();

    @BindString(R.string.arrival_short)
    String arrivalStr;

    @BindString(R.string.departure_short)
    String departureStr;


    @BindView(R.id.placeFrom)
    EditText placeFromInput;

    @BindView(R.id.placeTo)
    EditText placeToInput;

    @BindView(R.id.date_text)
    TextView dateText;

    @BindView(R.id.time_text)
    TextView timeText;

    @BindView(R.id.profile_spinner)
    Spinner profileSpinner;

    @BindView(R.id.query_appbar_layout)
    AppBarLayout appBarLayout;

    @BindView(R.id.frame_layout)
    View frameLayout;

    @BindView(R.id.connection_list_request_pending)
    View requestPendingView;

    @BindView(R.id.connection_list_query_incomplete)
    View queryIncompleteView;

    @BindView(R.id.connection_list_server_error)
    ServerErrorView serverErrorView;

    @BindView(R.id.connection_list)
    IntermodalJourneyListView journeyListView;

    @BindView(R.id.bottom_sheet_shadow)
    View bottomSheetShadow;

    private SupportMapFragment mapFragment;
    private View mapView;
    private GoogleMap map;
    private Map<String, ConnectionWrapper> polylineToConnection = new HashMap<>();

    private ArrayAdapter<SearchProfile> profileArrayAdapter;

    private List<ConnectionWrapper> recyclerVisibleConnections;
    private boolean recyclerVisibleStable = false;
    private LatLngBounds routeBounds;

    public IntermodalFragment() {
        // Required empty public constructor
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        query = new IntermodalQuery(searchProfiles);

        SharedPreferences prefs =
                getContext().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        query.load(prefs, PREFS_QUERY_PREFIX);

        connectionLoader = new IntermodalConnectionLoader(query);

        loadSavedInstanceState(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.intermodal_fragment, container, false);
        ButterKnife.bind(this, view);

        loadSavedInstanceState(savedInstanceState);

        journeyListView.init(connectionLoader);
        journeyListView.setLoadResultListener(this);
        journeyListView.setNestedScrollingEnabled(true);
        journeyListView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrollStateChanged(RecyclerView recyclerView, int newState) {
                if (newState == RecyclerView.SCROLL_STATE_IDLE) {
                    updateVisibleConnections();
                }
            }

            @Override
            public void onScrolled(RecyclerView recyclerView, int dx, int dy) {

            }

            private void updateVisibleConnections() {
                recyclerVisibleStable = true;
                List<ConnectionWrapper> conns = journeyListView.getVisibleConnections();
                if (recyclerVisibleConnections == null || !recyclerVisibleConnections.equals(conns)) {
                    StringBuilder sb = new StringBuilder();
                    for (ConnectionWrapper con : conns) {
                        sb.append(con.getId());
                        sb.append(" ");
                    }
                    Log.i(TAG, "JourneyListView scrolled: " + conns.size() +
                            "/" + journeyListView.getAllConnections().size() +
                            " visible connections: " + sb.toString());

                    recyclerVisibleConnections = conns;
                    updateMap(false);
                }
            }
        });

        profileArrayAdapter = new ArrayAdapter<>(
                getContext(), android.R.layout.simple_spinner_item, searchProfiles.profiles);
        profileArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        profileSpinner.setAdapter(profileArrayAdapter);

        mapFragment = (SupportMapFragment) getChildFragmentManager().findFragmentById(R.id.map);
        mapView = mapFragment.getView();

        MapBehavior mapBehavior = MapBehavior.from(frameLayout);
        mapBehavior.setTopView(appBarLayout);
        mapBehavior.setBottomView(journeyListView);

        final AnchoredBottomSheetBehavior bsBehavior = AnchoredBottomSheetBehavior.from(journeyListView);
        bsBehavior.setDragZoneHeight(getResources().getDimension(R.dimen.journey_list_floating_header_height));
        view.post(() -> {
            bsBehavior.setAnchorHeight((int) (view.getHeight() * .6));
        });

        bsBehavior.setBottomSheetCallback(new AnchoredBottomSheetBehavior.BottomSheetCallback() {
            @Override
            public void onStateChanged(@NonNull View bottomSheet, int newState) {
                if (newState == AnchoredBottomSheetBehavior.STATE_COLLAPSED ||
                        newState == AnchoredBottomSheetBehavior.STATE_HIDDEN) {
                    appBarLayout.setExpanded(true);
                } else {
                    appBarLayout.setExpanded(false);
                }
            }

            @Override
            public void onSlide(@NonNull View bottomSheet, float slideOffset) {

            }
        });

        updateVisibility();

        return view;
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        updateUiFromQuery();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        journeyListView.notifyDestroy();
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        this.context = context;
    }

    @Override
    public void onDetach() {
        super.onDetach();
        context = null;
    }

    @Override
    public void onSaveInstanceState(Bundle outState) {
//        query.updateBundle(outState);
    }

    protected void loadSavedInstanceState(Bundle savedInstanceState) {
        if (savedInstanceState != null) {
            // TODO
        }
    }

    private void saveQuery() {
        Log.i(TAG, "saveQuery");
        SharedPreferences prefs =
                getContext().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        query.save(prefs, PREFS_QUERY_PREFIX);
    }

    private void updateUiFromQuery() {
        if (query.getPlaceFrom() != null) {
            placeFromInput.setText(query.getPlaceFrom().name);
        } else {
            placeFromInput.setText("");
        }

        if (query.getPlaceTo() != null) {
            placeToInput.setText(query.getPlaceTo().name);
        } else {
            placeToInput.setText("");
        }

        if (query.getPPRSettings().pprSearchOptions != null) {
            profileSpinner.setSelection(profileArrayAdapter.getPosition(query.getPPRSettings().pprSearchOptions.profile));
        }

        Date d = query.getDateTime().getTime();
        updateTimeDisplay(query.isArrival(), d);
        updateDateDisplay(d);
    }

    @OnClick(R.id.placeFrom)
    public void onFromPlaceClick() {
        Log.i(TAG, "onFromPlaceClick");
        if (checkLocationPermission(FROM_LOCATION_PERMISSION_REQUEST)) {
            openPlacePicker(FROM_PLACE_PICKER_REQUEST, query.getPlaceFrom());
        }
    }

    @OnClick(R.id.placeTo)
    public void onToPlaceClick() {
        Log.i(TAG, "onToPlaceClick");
        if (checkLocationPermission(TO_LOCATION_PERMISSION_REQUEST)) {
            openPlacePicker(TO_PLACE_PICKER_REQUEST, query.getPlaceTo());
        }
    }

    private void openPlacePicker(int requestCode, NamedLocation startLocation) {
        PlacePicker.IntentBuilder builder = new PlacePicker.IntentBuilder();

        if (startLocation != null) {
            builder.setLatLngBounds(MapUtil.toBounds(startLocation.toLatLng(), 250));
        }

        try {
            startActivityForResult(builder.build(getActivity()), requestCode);
        } catch (GooglePlayServicesRepairableException | GooglePlayServicesNotAvailableException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.i(TAG, "onActivityResult: requestCode=" + requestCode + ", resultCode=" + resultCode);
        boolean queryUpdated = false;
        if (requestCode == FROM_PLACE_PICKER_REQUEST) {
            if (resultCode == RESULT_OK) {
                Place place = PlacePicker.getPlace(getActivity(), data);
                query.setPlaceFrom(new NamedLocation(place));
                placeFromInput.setText(query.getPlaceFrom().name);
                queryUpdated = true;
            }
        } else if (requestCode == TO_PLACE_PICKER_REQUEST) {
            if (resultCode == RESULT_OK) {
                Place place = PlacePicker.getPlace(getActivity(), data);
                query.setPlaceTo(new NamedLocation(place));
                placeToInput.setText(query.getPlaceTo().name);
                queryUpdated = true;
            }
        }

        if (queryUpdated) {
            saveQuery();
            sendSearchRequest();
        }
    }

    private boolean checkLocationPermission(int requestCode) {
        if (ContextCompat.checkSelfPermission(getActivity(),
                Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            requestPermissions(
                    new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                    requestCode);
            return false;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        if (grantResults.length > 0) {
            Log.i(TAG, "onRequestPermissionsResult: requestCode=" + requestCode +
                    ", permission=" + permissions[0] +
                    ", grantResult=" + grantResults[0]);
        } else {
            Log.i(TAG, "onRequestPermissionsResult: requestCode=" + requestCode +
                    ", empty result");
        }

        if (requestCode == FROM_LOCATION_PERMISSION_REQUEST) {
            openPlacePicker(FROM_PLACE_PICKER_REQUEST, query.getPlaceFrom());
        } else if (requestCode == TO_LOCATION_PERMISSION_REQUEST) {
            openPlacePicker(TO_PLACE_PICKER_REQUEST, query.getPlaceTo());
        }
    }

    @OnClick(R.id.switch_places_btn)
    void swapStartDest() {
        query.swapStartDest();

        updateUiFromQuery();

        saveQuery();
        sendSearchRequest();
    }

    @OnClick(R.id.date_select)
    void showDatePickerDialog() {
        DialogFragment dialogFragment = DatePickerDialogFragment.newInstance(
                query.getYear(), query.getMonth(), query.getDay());
        dialogFragment.setTargetFragment(this, 0);
        dialogFragment.show(getActivity().getSupportFragmentManager(), "datePicker");
    }

    @OnClick(R.id.time_select)
    void showTimePickerDialog() {
        DialogFragment dialogFragment = TimePickerDialogFragment.newInstance(
                query.isArrival(), query.getHour(), query.getMinute());
        dialogFragment.setTargetFragment(this, 0);
        dialogFragment.show(getActivity().getSupportFragmentManager(), "timePicker");
    }

    @Override
    public void onDateSet(@Nullable DatePicker view, int year, int month, int day) {
        query.setDate(year, month, day);

        updateDateDisplay(query.getDateTime().getTime());

        saveQuery();
        sendSearchRequest();
    }

    @Override
    public void onTimeSet(boolean isArrival, int hour, int minute) {
        query.setTime(isArrival, hour, minute);

        updateTimeDisplay(isArrival, query.getDateTime().getTime());

        saveQuery();
        sendSearchRequest();
    }

    @OnItemSelected(R.id.profile_spinner)
    public void onProfileSelected(AdapterView<?> parent, View view, int position, long id) {
        SearchProfile profile = (SearchProfile) parent.getItemAtPosition(position);
        Log.i(TAG, "Selected Search Profile: " + profile);
        query.getPPRSettings().setProfile(profile);
        Status.get().setPprSearchOptions(query.getPPRSettings().pprSearchOptions);

        saveQuery();
        sendSearchRequest();
    }

    private void updateTimeDisplay(boolean isArrival, Date time) {
        String formattedTime = TimeUtil.formatTime(time);
        String formattedArrival = (isArrival ? arrivalStr : departureStr);
        timeText.setText(formattedArrival + " " + formattedTime);
    }

    private void updateDateDisplay(Date date) {
        dateText.setText(TimeUtil.formatDate(date));
    }

    private void updateVisibility() {
        if (!query.isComplete()) {
            journeyListView.setVisibility(View.GONE);
            mapView.setVisibility(View.GONE);
            bottomSheetShadow.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.GONE);
            queryIncompleteView.setVisibility(View.VISIBLE);
            serverErrorView.setVisibility(View.GONE);
            return;
        }

        if (connectionLoader.isInitialRequestPending()) {
            journeyListView.setVisibility(View.GONE);
            mapView.setVisibility(View.GONE);
            bottomSheetShadow.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.VISIBLE);
            queryIncompleteView.setVisibility(View.GONE);
            serverErrorView.setVisibility(View.GONE);
            return;
        }

        if (connectionLoader.isServerError()) {
            journeyListView.setVisibility(View.GONE);
            mapView.setVisibility(View.GONE);
            bottomSheetShadow.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.GONE);
            queryIncompleteView.setVisibility(View.GONE);
            serverErrorView.setVisibility(View.VISIBLE);
            return;
        }

        journeyListView.setVisibility(View.VISIBLE);
        mapView.setVisibility(View.VISIBLE);
        bottomSheetShadow.setVisibility(View.VISIBLE);
        requestPendingView.setVisibility(View.GONE);
        queryIncompleteView.setVisibility(View.GONE);
        serverErrorView.setVisibility(View.GONE);
    }

    private void sendSearchRequest() {
        Log.i(TAG, "sendSearchRequest");
        if (query.isComplete()) {
            recyclerVisibleStable = false;
            journeyListView.loadInitial();
        }
    }

    protected void tryToUpdateMap(boolean setCamera) {
        try {
            Log.i(TAG, "tryToUpdateMap");
            getActivity().runOnUiThread(() -> {
                new OnMapAndViewReadyListener(mapFragment, googleMap -> {
                    this.map = googleMap;
                    updateMap(setCamera);
                });
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected void updateMap(boolean setCamera) {
        synchronized (map) {
            Log.i(TAG, "updateMap");
            map.clear();
            List<ConnectionWrapper> connections = recyclerVisibleConnections;
            if (!recyclerVisibleStable || connections == null || connections.isEmpty()) {
                connections = connectionLoader.getData();
            }
            if (connections.isEmpty() || !query.isComplete()) {
                return;
            }

            LatLng start = query.getPlaceFrom().toLatLng();
            LatLng destination = query.getPlaceTo().toLatLng();
            LatLngBounds.Builder boundsBuilder = LatLngBounds.builder();
            boundsBuilder.include(start).include(destination);

            map.addMarker(new MarkerOptions()
                    .position(start)
                    .title(getResources().getString(R.string.start)));
            map.addMarker(new MarkerOptions()
                    .position(destination)
                    .title(getResources().getString(R.string.destination))
                    .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE)));

            polylineToConnection.clear();
            for (ConnectionWrapper connection : connections) {
                Iterable<LatLng> routePath = connection.getPath();
                Polyline polyline = map.addPolyline(new PolylineOptions()
                        .addAll(routePath)
                        .color(connection.getColor(getActivity()))
                        .clickable(true)
                        .width(20));
                polylineToConnection.put(polyline.getId(), connection);
                for (LatLng pt : routePath) {
                    boundsBuilder.include(pt);
                }
            }

            if (setCamera) {
                routeBounds = boundsBuilder.build();
                mapView.post(this::setMapCameraBounds);
            }

            map.setOnPolylineClickListener(this);
        }
    }

    protected void setMapCameraBounds() {
        if (routeBounds == null || mapView.getHeight() == 0) {
            return;
        }
        try {
            map.moveCamera(CameraUpdateFactory.newLatLngBounds(routeBounds, 200));
        } catch (IllegalStateException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void onPolylineClick(Polyline polyline) {
        ConnectionWrapper connection = polylineToConnection.get(polyline.getId());
        if (connection != null) {
            Status.get().setConnection(connection);
            Status.get().setPprSearchOptions(query.getPPRSettings().pprSearchOptions);
            Intent intent = new Intent(getContext(), IntermodalDetailActivity.class);
            startActivity(intent);
        } else {
            Log.w(TAG, "Unknown polyline clicked: " + polyline.getId());
        }
    }

    @Override
    public void loading(BaseJourneyListView.LoadType type) {
        if (type == BaseJourneyListView.LoadType.Initial) {
//            appBarLayout.setExpanded(true, true);
            AnchoredBottomSheetBehavior bottomSheetBehavior = AnchoredBottomSheetBehavior.from(journeyListView);
            bottomSheetBehavior.setState(AnchoredBottomSheetBehavior.STATE_COLLAPSED);
            updateVisibility();
        }
    }

    @Override
    public void loaded(BaseJourneyListView.LoadType type, RoutingResponse res, int newConnectionCount) {
        if (type == BaseJourneyListView.LoadType.Initial && newConnectionCount == 0) {
            serverErrorView.setEmptyResponse();
        }
        updateVisibility();
        tryToUpdateMap(true);
    }

    @Override
    public void failed(BaseJourneyListView.LoadType type, Throwable t) {
        if (type == BaseJourneyListView.LoadType.Initial && t instanceof MotisErrorException) {
            serverErrorView.setErrorCode((MotisErrorException) t);
        }
        updateVisibility();
    }

    @Override
    public void routeLoaded(WalkKey key, RouteWrapper route) {
        updateMap(false);
    }

    @Override
    public void routeRequestFailed(WalkKey key, Throwable t) {

    }

    @Override
    public void serverChanged() {
        Log.i(TAG, "serverChanged");
        sendSearchRequest();
    }
}
