package de.motis_project.app2.query;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.text.format.DateFormat;
import android.view.LayoutInflater;
import android.view.View;
import android.view.Window;
import android.widget.RadioButton;
import android.widget.TimePicker;

import androidx.fragment.app.DialogFragment;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnCheckedChanged;
import de.motis_project.app2.R;

public class TimePickerDialogFragment extends DialogFragment
        implements DialogInterface.OnClickListener, TimePicker.OnTimeChangedListener {

    public interface ChangeListener {
        void onTimeSet(boolean isArrival, int hour, int minute);
    }

    private static final String IS_ARRIVAL = "TIME_PICKER_IS_ARRIVAL";
    private static final String HOUR = "TIME_PICKER_HOUR";
    private static final String MINUTE = "TIME_PICKER_MINUTE";

    private Bundle state = new Bundle();

    @BindView(R.id.depature_btn)
    RadioButton depBtn;

    @BindView(R.id.arrival_btn)
    RadioButton arrBtn;

    @BindView(R.id.time_picker)
    TimePicker timePicker;

    public static TimePickerDialogFragment newInstance(boolean isArrival, int hour, int minute) {
        TimePickerDialogFragment fragment = new TimePickerDialogFragment();

        Bundle arguments = new Bundle();
        arguments.putBoolean(IS_ARRIVAL, isArrival);
        arguments.putInt(HOUR, hour);
        arguments.putInt(MINUTE, minute);
        fragment.setArguments(arguments);

        return fragment;
    }

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {
        LayoutInflater inflater = getActivity().getLayoutInflater();
        View v = inflater.inflate(R.layout.query_dialog_time_picker, null);
        ButterKnife.bind(this, v);

        initFromBundle(savedInstanceState != null ? savedInstanceState : getArguments());
        setArrival();
        setTime();

        AlertDialog dialog =
                new AlertDialog.Builder(getActivity())
                        .setView(v)
                        .setPositiveButton(android.R.string.ok, this)
                        .setNegativeButton(android.R.string.cancel, null)
                        .create();
        dialog.getWindow().requestFeature(Window.FEATURE_NO_TITLE);
        return dialog;
    }

    @Override
    public void onSaveInstanceState(Bundle outState) {
        outState.putBoolean(IS_ARRIVAL, state.getBoolean(IS_ARRIVAL));
        outState.putInt(HOUR, state.getInt(HOUR));
        outState.putInt(MINUTE, state.getInt(MINUTE));
    }

    private void initFromBundle(Bundle b) {
        this.state = b;
    }

    private void setArrival() {
        if (state.getBoolean(IS_ARRIVAL)) {
            arrBtn.setChecked(true);
            depBtn.setChecked(false);
        } else {
            arrBtn.setChecked(false);
            depBtn.setChecked(true);
        }
    }

    private void setTime() {
        int hour = state.getInt(HOUR);
        int minute = state.getInt(MINUTE);

        timePicker.setIs24HourView(DateFormat.is24HourFormat(getContext()));
        timePicker.setOnTimeChangedListener(this);
        timePicker.setCurrentHour(hour);
        timePicker.setCurrentMinute(minute);
    }

    private ChangeListener getListener() {
        return (ChangeListener) getTargetFragment();
    }

    @OnCheckedChanged(R.id.arrival_btn)
    void onChecked(boolean checked) {
        state.putBoolean(IS_ARRIVAL, checked);
    }

    @Override
    public void onTimeChanged(TimePicker timePicker, int hour, int minute) {
        state.putInt(HOUR, hour);
        state.putInt(MINUTE, minute);
    }

    @Override
    public void onClick(DialogInterface dialogInterface, int i) {
        getListener().onTimeSet(state.getBoolean(IS_ARRIVAL),
                                state.getInt(HOUR),
                                state.getInt(MINUTE));
    }
}
