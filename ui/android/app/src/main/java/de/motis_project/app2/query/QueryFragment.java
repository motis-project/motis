package de.motis_project.app2.query;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;

import com.google.android.material.appbar.AppBarLayout;

import java.util.Calendar;
import java.util.Date;

import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import butterknife.OnTouch;
import de.motis_project.app2.R;
import de.motis_project.app2.ServerChangeListener;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.io.error.MotisErrorException;
import de.motis_project.app2.journey.BaseJourneyListView;
import de.motis_project.app2.journey.JourneyListView;
import de.motis_project.app2.journey.ServerErrorView;
import de.motis_project.app2.query.guesser.GuesserActivity;
import motis.routing.RoutingResponse;

public class QueryFragment extends Fragment
        implements android.app.DatePickerDialog.OnDateSetListener,
        TimePickerDialogFragment.ChangeListener,
        BaseJourneyListView.LoadResultListener, ServerChangeListener {
    private static final String TAG = "QueryFragment";

    public static final int SELECT_START_LOCATION = 1;
    public static final int SELECT_DEST_LOCATION = 2;

    private Context context;
    private Query query;
    private RoutingConnectionLoader connectionLoader;

    @BindString(R.string.arrival_short)
    String arrivalStr;

    @BindString(R.string.departure_short)
    String departureStr;

    @BindView(R.id.start_input)
    EditText fromInput;

    @BindView(R.id.dest_input)
    EditText toInput;

    @BindView(R.id.date_text)
    TextView dateText;

    @BindView(R.id.time_text)
    TextView timeText;

    @BindView(R.id.connection_list)
    JourneyListView journeyListView;

    @BindView(R.id.connection_list_request_pending)
    View requestPendingView;

    @BindView(R.id.connection_list_query_incomplete)
    View queryIncompleteView;

    @BindView(R.id.connection_list_server_error)
    ServerErrorView serverErrorView;

    @BindView(R.id.query_appbar_layout)
    AppBarLayout appBarLayout;

    public QueryFragment() {
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.query_fragment, container, false);
        ButterKnife.bind(this, view);

        query = new Query(
                savedInstanceState,
                getContext().getSharedPreferences("route", Context.MODE_PRIVATE));
        connectionLoader = new RoutingConnectionLoader(query);
        journeyListView.init(connectionLoader);
        journeyListView.setLoadResultListener(this);

        Date d = query.getTime();
        updateTimeDisplay(query.isArrival(), d);
        updateDateDisplay(d);

        fromInput.setText(query.getFromName());
        toInput.setText(query.getToName());

        sendSearchRequest();

        return view;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        journeyListView.notifyDestroy();
        System.out.println("QueryFragment.onDestroyView");
    }

    @Override
    public void onSaveInstanceState(Bundle outState) {
        System.out.println("QueryFragment.onSaveInstanceState");
        if (query != null) {
            query.updateBundle(outState);
        }
    }

    @OnTouch(R.id.start_input)
    boolean openSearchStart(View v, MotionEvent e) {
        openGuesser(v, e, SELECT_START_LOCATION, fromInput.getText().toString());
        return true;
    }

    @OnTouch(R.id.dest_input)
    boolean openSearchDest(View v, MotionEvent e) {
        openGuesser(v, e, SELECT_DEST_LOCATION, toInput.getText().toString());
        return true;
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

    @OnClick(R.id.switch_stations_btn)
    void swapStations() {
        query.swapStations();

        String from = fromInput.getText().toString();
        fromInput.setText(toInput.getText().toString());
        toInput.setText(from);

        sendSearchRequest();
    }

    private void openGuesser(@Nullable View v, MotionEvent e, int requestCode, String query) {
        if (e.getAction() != MotionEvent.ACTION_UP || context == null) {
            return;
        }

        Intent i = new Intent(context, GuesserActivity.class);
        i.putExtra(GuesserActivity.QUERY, query);
        startActivityForResult(i, requestCode);
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
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (data == null || data.getExtras() == null ||
                data.getExtras().getString(GuesserActivity.RESULT_NAME) == null ||
                data.getExtras().getString(GuesserActivity.RESULT_ID) == null ||
                resultCode != Activity.RESULT_OK) {
            return;
        }

        String name = data.getExtras().getString(GuesserActivity.RESULT_NAME);
        String id = data.getExtras().getString(GuesserActivity.RESULT_ID);
        switch (requestCode) {
            case SELECT_START_LOCATION:
                query.setFrom(id, name);
                fromInput.setText(name);
                sendSearchRequest();
                break;
            case SELECT_DEST_LOCATION:
                query.setTo(id, name);
                toInput.setText(name);
                sendSearchRequest();
                break;
        }
    }

    @Override
    public void onDateSet(@Nullable DatePicker view, int year, int month, int day) {
        query.setDate(year, month, day);

        Calendar cal = Calendar.getInstance();
        Query.setDate(cal, year, month, day);
        updateDateDisplay(cal.getTime());

        sendSearchRequest();
    }

    @Override
    public void onTimeSet(boolean isArrival, int hour, int minute) {
        query.setTime(isArrival, hour, minute);

        Calendar cal = Calendar.getInstance();
        Query.setTime(cal, hour, minute);
        updateTimeDisplay(isArrival, cal.getTime());

        sendSearchRequest();
    }

    private void sendSearchRequest() {
        journeyListView.loadInitial();
    }

    private void updateTimeDisplay(boolean isArrival, Date time) {
        String formattedTime = TimeUtil.formatTime(time);
        String formattedArrival = (isArrival ? arrivalStr : departureStr);
        timeText.setText(formattedArrival + " " + formattedTime);
    }

    private void updateDateDisplay(Date date) {
        dateText.setText(TimeUtil.formatDate(date));
    }

    protected void updateVisibility() {
        if (!query.isComplete()) {
            journeyListView.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.GONE);
            queryIncompleteView.setVisibility(View.VISIBLE);
            serverErrorView.setVisibility(View.GONE);
            return;
        }

        if (connectionLoader.isInitialRequestPending()) {
            journeyListView.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.VISIBLE);
            queryIncompleteView.setVisibility(View.GONE);
            serverErrorView.setVisibility(View.GONE);
            return;
        }

        if (connectionLoader.isServerError()) {
            journeyListView.setVisibility(View.GONE);
            requestPendingView.setVisibility(View.GONE);
            queryIncompleteView.setVisibility(View.GONE);
            serverErrorView.setVisibility(View.VISIBLE);
            return;
        }

        journeyListView.setVisibility(View.VISIBLE);
        requestPendingView.setVisibility(View.GONE);
        queryIncompleteView.setVisibility(View.GONE);
        serverErrorView.setVisibility(View.GONE);
    }

    @Override
    public void loading(BaseJourneyListView.LoadType type) {
        if (type == BaseJourneyListView.LoadType.Initial) {
            appBarLayout.setExpanded(true, true);
            updateVisibility();
        }
    }

    @Override
    public void loaded(BaseJourneyListView.LoadType type, RoutingResponse res, int newConnectionCount) {
        if (type == BaseJourneyListView.LoadType.Initial && newConnectionCount == 0) {
            serverErrorView.setEmptyResponse();
        }
        updateVisibility();
    }

    @Override
    public void failed(BaseJourneyListView.LoadType type, Throwable t) {
        if (type == BaseJourneyListView.LoadType.Initial && t instanceof MotisErrorException) {
            serverErrorView.setErrorCode((MotisErrorException) t);
        }
        updateVisibility();
    }

    @Override
    public void serverChanged() {
        Log.i(TAG, "serverChanged");
        sendSearchRequest();
    }
}
