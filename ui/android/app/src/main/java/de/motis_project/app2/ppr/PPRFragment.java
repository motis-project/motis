package de.motis_project.app2.ppr;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.design.widget.AppBarLayout;
import android.support.v4.app.Fragment;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;

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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import butterknife.OnItemSelected;
import de.motis_project.app2.R;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.io.error.MotisErrorException;
import de.motis_project.app2.lib.AnchoredBottomSheetBehavior;
import de.motis_project.app2.lib.MapBehavior;
import de.motis_project.app2.ppr.profiles.SearchProfile;
import de.motis_project.app2.ppr.profiles.SearchProfiles;
import de.motis_project.app2.ppr.query.PPRQuery;
import de.motis_project.app2.ppr.route.RouteWrapper;
import motis.ppr.FootRoutingResponse;
import motis.ppr.Route;
import motis.ppr.Routes;
import rx.Subscription;
import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Action1;
import rx.internal.util.SubscriptionList;
import rx.schedulers.Schedulers;

import static android.app.Activity.RESULT_OK;


public class PPRFragment extends Fragment implements GoogleMap.OnPolylineClickListener {

    private static final String TAG = "PPRFragment";
    private static final String KEY_PLACE_FROM = "ppr.placeFrom";
    private static final String KEY_PLACE_TO = "ppr.placeTo";
    private static final String KEY_PROFILE = "ppr.pprSearchOptions.profile.id";
    private static final String KEY_MAX_DURATION = "ppr.pprSearchOptions.maxDuration";
    private static final String PREFS_NAME = "ppr";
    private static final String PREFS_QUERY_PREFIX = "query.";

    private static final int FROM_PLACE_PICKER_REQUEST = 1;
    private static final int TO_PLACE_PICKER_REQUEST = 2;
    private static final int FROM_LOCATION_PERMISSION_REQUEST = 1;
    private static final int TO_LOCATION_PERMISSION_REQUEST = 2;

    private Context context;

    private final SearchProfiles searchProfiles = new SearchProfiles();

    private PPRQuery query;
    private boolean serverError = false;
    private boolean requestPending = true;
    private SubscriptionList subscriptions = new SubscriptionList();
    private final List<RouteWrapper> routeList = new ArrayList<>();
    private final RouteSummaryAdapter routeSummaryAdapter = new RouteSummaryAdapter(routeList);

    @BindView(R.id.pprPlaceFrom)
    EditText placeFromInput;

    @BindView(R.id.pprPlaceTo)
    EditText placeToInput;

    @BindView(R.id.route_list)
    RouteListView routeListView;

    @BindView(R.id.route_list_request_pending)
    View requestPendingView;

    @BindView(R.id.route_list_query_incomplete)
    View queryIncompleteView;

    @BindView(R.id.route_list_server_error)
    ServerErrorView serverErrorView;

    @BindView(R.id.ppr_appbar_layout)
    AppBarLayout appBarLayout;

    @BindView(R.id.profile_spinner)
    Spinner profileSpinner;

    @BindView(R.id.frame_layout)
    View frameLayout;

    @BindView(R.id.bottom_sheet_shadow)
    View bottomSheetShadow;

    private SupportMapFragment mapFragment;
    private View mapView;
    private GoogleMap map;
    private Map<String, RouteWrapper> polylineToRoute = new HashMap<>();
    private LatLngBounds routeBounds;

    private ArrayAdapter<SearchProfile> profileArrayAdapter;

    public PPRFragment() {
        // Required empty public constructor
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");

        query = new PPRQuery(searchProfiles);

        SharedPreferences prefs =
                getContext().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        query.load(prefs, PREFS_QUERY_PREFIX);

        loadSavedInstanceState(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        Log.i(TAG, "onCreateView");
        // Inflate the layout for this fragment
        final View view = inflater.inflate(R.layout.ppr_fragment, container, false);
        ButterKnife.bind(this, view);

        loadSavedInstanceState(savedInstanceState);

        profileArrayAdapter = new ArrayAdapter<>(
                getContext(), android.R.layout.simple_spinner_item, searchProfiles.profiles);
        profileArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        profileSpinner.setAdapter(profileArrayAdapter);

        mapFragment = (SupportMapFragment) getChildFragmentManager().findFragmentById(R.id.map);
        mapView = mapFragment.getView();

        routeListView.setAdapter(routeSummaryAdapter);

        MapBehavior mapBehavior = MapBehavior.from(frameLayout);
        mapBehavior.setTopView(appBarLayout);
        mapBehavior.setBottomView(routeListView);

        final AnchoredBottomSheetBehavior bsBehavior = AnchoredBottomSheetBehavior.from(routeListView);
        float density = getResources().getDisplayMetrics().density;
        bsBehavior.setDragZoneHeight(60 * density);
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

    protected void loadSavedInstanceState(Bundle savedInstanceState) {
        if (savedInstanceState != null) {
            query.placeFrom = savedInstanceState.getParcelable(KEY_PLACE_FROM);
            query.placeTo = savedInstanceState.getParcelable(KEY_PLACE_TO);
            query.pprSearchOptions.profile =
                    searchProfiles.getById(
                            savedInstanceState.getString(KEY_PROFILE, query.pprSearchOptions.profile.getId()),
                            query.pprSearchOptions.profile);
            query.pprSearchOptions.maxDuration =
                    savedInstanceState.getInt(KEY_MAX_DURATION, query.pprSearchOptions.maxDuration);
        }
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        Log.i(TAG, "onViewCreated");

        updateUiFromQuery();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        subscriptions.clear();
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        this.context = context;
    }

    @Override
    public void onDetach() {
        super.onDetach();
        this.context = null;
    }

    @OnClick(R.id.switch_ppr_places_btn)
    void swapStartDest() {
        query.swapStartDest();

        updateUiFromQuery();

        saveQuery();
        sendSearchRequest();
    }

    private void updateUiFromQuery() {
        if (query.placeFrom != null) {
            placeFromInput.setText(query.placeFrom.name);
        } else {
            placeFromInput.setText("");
        }

        if (query.placeTo != null) {
            placeToInput.setText(query.placeTo.name);
        } else {
            placeToInput.setText("");
        }

        if (query.pprSearchOptions != null) {
            profileSpinner.setSelection(profileArrayAdapter.getPosition(query.pprSearchOptions.profile));
        }
    }

    @OnItemSelected(R.id.profile_spinner)
    public void onProfileSelected(AdapterView<?> parent, View view, int position, long id) {
        SearchProfile profile = (SearchProfile) parent.getItemAtPosition(position);
        Log.i(TAG, "Selected Search Profile: " + profile);
        query.pprSearchOptions.profile = profile;
        saveQuery();
        sendSearchRequest();
    }

    @Override
    public void onSaveInstanceState(Bundle outState) {
        Log.i(TAG, "onSaveInstanceState");
        super.onSaveInstanceState(outState);
        outState.putParcelable(KEY_PLACE_FROM, query.placeFrom);
        outState.putParcelable(KEY_PLACE_TO, query.placeTo);
        if (query.pprSearchOptions != null) {
            outState.putString(KEY_PROFILE, query.pprSearchOptions.profile.getId());
            outState.putInt(KEY_MAX_DURATION, query.pprSearchOptions.maxDuration);
        }
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        Log.i(TAG, "onActivityCreated");
    }

    private void saveQuery() {
        Log.i(TAG, "saveQuery");
        SharedPreferences prefs =
                getContext().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        query.save(prefs, PREFS_QUERY_PREFIX);
    }

    @OnClick(R.id.pprPlaceFrom)
    public void onFromPlaceClick() {
        Log.i(TAG, "onFromPlaceClick");
        if (checkLocationPermission(FROM_LOCATION_PERMISSION_REQUEST)) {
            openPlacePicker(FROM_PLACE_PICKER_REQUEST, query.placeFrom);
        }
    }

    @OnClick(R.id.pprPlaceTo)
    public void onToPlaceClick() {
        Log.i(TAG, "onToPlaceClick");
        if (checkLocationPermission(TO_LOCATION_PERMISSION_REQUEST)) {
            openPlacePicker(TO_PLACE_PICKER_REQUEST, query.placeTo);
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
                query.placeFrom = new NamedLocation(place);
                placeFromInput.setText(query.placeFrom.name);
                queryUpdated = true;
            }
        } else if (requestCode == TO_PLACE_PICKER_REQUEST) {
            if (resultCode == RESULT_OK) {
                Place place = PlacePicker.getPlace(getActivity(), data);
                query.placeTo = new NamedLocation(place);
                placeToInput.setText(query.placeTo.name);
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
            openPlacePicker(FROM_PLACE_PICKER_REQUEST, query.placeFrom);
        } else if (requestCode == TO_LOCATION_PERMISSION_REQUEST) {
            openPlacePicker(TO_PLACE_PICKER_REQUEST, query.placeTo);
        }
    }

    private void updateVisibility() {
        if (!query.isComplete()) {
            routeListView.setVisibility(View.GONE);
            mapView.setVisibility(View.GONE);
            bottomSheetShadow.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.GONE);
            queryIncompleteView.setVisibility(View.VISIBLE);
            serverErrorView.setVisibility(View.GONE);
            return;
        }

        if (requestPending) {
            routeListView.setVisibility(View.GONE);
            mapView.setVisibility(View.GONE);
            bottomSheetShadow.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.VISIBLE);
            queryIncompleteView.setVisibility(View.GONE);
            serverErrorView.setVisibility(View.GONE);
            return;
        }

        if (serverError) {
            routeListView.setVisibility(View.GONE);
            mapView.setVisibility(View.GONE);
            bottomSheetShadow.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.GONE);
            queryIncompleteView.setVisibility(View.GONE);
            serverErrorView.setVisibility(View.VISIBLE);
            return;
        }

        routeListView.setVisibility(View.VISIBLE);
        mapView.setVisibility(View.VISIBLE);
        bottomSheetShadow.setVisibility(View.VISIBLE);
        requestPendingView.setVisibility(View.GONE);
        queryIncompleteView.setVisibility(View.GONE);
        serverErrorView.setVisibility(View.GONE);

        AnchoredBottomSheetBehavior bottomSheetBehavior = AnchoredBottomSheetBehavior.from(routeListView);
        bottomSheetBehavior.setState(AnchoredBottomSheetBehavior.STATE_COLLAPSED);
    }

    private void sendSearchRequest() {
        Log.i(TAG, "sendSearchRequest");
        appBarLayout.setExpanded(true, true);

        subscriptions.unsubscribe();
        subscriptions = new SubscriptionList();

        serverError = false;
        requestPending = true;
        clearRoutes();

        updateVisibility();

        if (!query.isComplete()) {
            return;
        }

        route(resObj -> {
            FootRoutingResponse res = (FootRoutingResponse) resObj;

            Log.i(TAG, "Received FootRoutingResponse");

            requestPending = false;

            if (res.routesLength() == 1) {
                Routes routes = res.routes(0);
                if (routes.routesLength() == 0) {
                    serverError = true;
                    serverErrorView.setEmptyResponse();
                }
                setRoutes(routes);
            } else {
                serverError = true;
                serverErrorView.setEmptyResponse();
            }
            updateVisibility();
            tryToUpdateMap();
        }, t -> {
            Log.w(TAG, "PPR Request failed");
            requestPending = false;
            serverError = true;
            if (t instanceof MotisErrorException) {
                MotisErrorException mee = (MotisErrorException) t;
                Log.w(TAG, "Motis Error: " + mee.category + ": " + mee.reason + " (" + mee.code + ")");
                serverErrorView.setErrorCode(mee);
            }
            updateVisibility();
        });
    }


    public void route(Action1 action, Action1<Throwable> errorAction) {
        Subscription sub = Status.get().getServer()
                .pprRoute(query)
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(action, errorAction);
        subscriptions.add(sub);
    }


    protected void clearRoutes() {
        routeList.clear();
        routeSummaryAdapter.notifyDataSetChanged();
    }

    protected void setRoutes(Routes routes) {
        routeList.clear();
        ArrayList<Route> rawRoutes = new ArrayList<>(routes.routesLength());
        for (int i = 0; i < routes.routesLength(); i++) {
            rawRoutes.add(routes.routes(i));
        }
        Collections.sort(rawRoutes, (a, b) -> Double.compare(a.durationExact(), b.durationExact()));
        for (int i = 0; i < rawRoutes.size(); i++) {
            routeList.add(new RouteWrapper(rawRoutes.get(i), i));
        }
        routeSummaryAdapter.notifyDataSetChanged();
    }

    protected void tryToUpdateMap() {
        try {
            Log.i(TAG, "tryToUpdateMap");
            getActivity().runOnUiThread(() -> {
                new OnMapAndViewReadyListener(mapFragment, googleMap -> {
                    this.map = googleMap;
                    updateMap();
                });
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected void updateMap() {
        Log.i(TAG, "updateMap");
        map.clear();
        if (routeList.isEmpty() || !query.isComplete()) {
            return;
        }

        LatLng start = query.placeFrom.toLatLng();
        LatLng destination = query.placeTo.toLatLng();
        LatLngBounds.Builder boundsBuilder = LatLngBounds.builder();
        boundsBuilder.include(start).include(destination);

        map.addMarker(new MarkerOptions()
                .position(start)
                .title(getResources().getString(R.string.start)));
        map.addMarker(new MarkerOptions()
                .position(destination)
                .title(getResources().getString(R.string.destination))
                .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE)));

        polylineToRoute.clear();
        for (RouteWrapper route : routeList) {
            Iterable<LatLng> routePath = route.getPath();
            Polyline polyline = map.addPolyline(new PolylineOptions()
                    .addAll(routePath)
                    .color(route.getColor(getActivity()))
                    .clickable(true)
                    .width(20));
            polylineToRoute.put(polyline.getId(), route);
            for (LatLng pt : routePath) {
                boundsBuilder.include(pt);
            }
        }

        routeBounds = boundsBuilder.build();
        mapView.post(this::setMapCameraBounds);

        map.setOnPolylineClickListener(this);
    }

    protected void setMapCameraBounds() {
        Log.i(TAG, "Map View size: " + mapView.getWidth() + "x" + mapView.getHeight());
        View frameLayout = getView().findViewById(R.id.frame_layout);
        Log.i(TAG, "Frame View size: " + frameLayout.getWidth() + "x" + frameLayout.getHeight());

        if (routeBounds == null) {
            return;
        }

        map.moveCamera(CameraUpdateFactory.newLatLngBounds(routeBounds, 200));
    }

    @Override
    public void onPolylineClick(Polyline polyline) {
        RouteWrapper route = polylineToRoute.get(polyline.getId());
        if (route != null) {
            Status.get().setPprRoute(route);
            Intent intent = new Intent(getContext(), RouteDetailActivity.class);
            startActivity(intent);
        } else {
            Log.w(TAG, "Unknown polyline clicked: " + polyline.getId());
        }
    }
}
