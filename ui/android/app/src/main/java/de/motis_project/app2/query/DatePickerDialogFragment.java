package de.motis_project.app2.query;

import android.app.DatePickerDialog;
import android.app.Dialog;
import android.os.Bundle;

import androidx.fragment.app.DialogFragment;

public class DatePickerDialogFragment extends DialogFragment {
    private static final String YEAR = "YEAR";
    private static final String MONTH = "MONTH";
    private static final String DAY = "DAY";

    public static DatePickerDialogFragment newInstance(int year, int month, int day) {
        DatePickerDialogFragment datePicker = new DatePickerDialogFragment();

        Bundle args = new Bundle();
        args.putInt(YEAR, year);
        args.putInt(MONTH, month);
        args.putInt(DAY, day);
        datePicker.setArguments(args);

        return datePicker;
    }

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {
        Bundle args = getArguments();
        return new android.app.DatePickerDialog(getActivity(), getListener(),
                                                args.getInt(YEAR),
                                                args.getInt(MONTH),
                                                args.getInt(DAY));
    }

    private DatePickerDialog.OnDateSetListener getListener() {
        return (DatePickerDialog.OnDateSetListener) getTargetFragment();
    }
}
