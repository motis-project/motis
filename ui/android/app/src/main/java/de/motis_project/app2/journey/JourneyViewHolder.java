package de.motis_project.app2.journey;

import android.view.LayoutInflater;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.recyclerview.widget.RecyclerView;

import butterknife.ButterKnife;

public class JourneyViewHolder extends RecyclerView.ViewHolder {
    final LayoutInflater inflater;

    public JourneyViewHolder(View view, @Nullable LayoutInflater inflater) {
        super(view);
        this.inflater = inflater;
        ButterKnife.bind(this, view);
    }
}
