package de.motis_project.app2.detail;

import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;

import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.journey.ConnectionWrapper;
import de.motis_project.app2.journey.CopyConnection;
import motis.Stop;

public class DetailActivity extends AppCompatActivity implements DetailClickHandler {
    public static final String SHOW_SAVE_ACTION = "SHOW_SAVE_ACTION";

    private ConnectionWrapper con;
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        requestWindowFeature(Window.FEATURE_ACTION_BAR);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.detail);
        setSupportActionBar((Toolbar) findViewById(R.id.journey_detail_toolbar));
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        con = Status.get().getConnection();

        String formattedDate = SimpleDateFormat
                .getDateInstance(java.text.DateFormat.SHORT)
                .format(new Date(con.getFirstStop().departure().scheduleTime() * 1000));
        setTitle(formattedDate);

        ButterKnife.bind(this);
        initHeader();
        create();
    }

    void create() {
        TransportBuilder.setConnection(getLayoutInflater(),
                journeyDetails, con, expandedSections);
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

    }
}
