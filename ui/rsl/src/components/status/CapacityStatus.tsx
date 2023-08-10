import { useQuery } from "@tanstack/react-query";
import { add, getUnixTime } from "date-fns";
import { useAtom } from "jotai";
import { Check, ChevronsUpDown } from "lucide-react";
import React, { ReactElement, useState } from "react";

import {
  PaxMonCapacityStats,
  PaxMonCapacityStatusRequest,
  PaxMonCapacityStatusResponse,
  PaxMonProviderCapacityStats,
  PaxMonProviderInfo,
} from "@/api/protocol/motis/paxmon";

import { useLookupScheduleInfoQuery } from "@/api/lookup";
import { queryKeys, sendPaxMonCapacityStatusRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber, formatPercent } from "@/data/numberFormat";

import { getScheduleRange } from "@/util/scheduleRange";

import DatePicker from "@/components/inputs/DatePicker";
import DetailedCapacityStatus from "@/components/status/DetailedCapacityStatus";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";

import { cn } from "@/lib/utils";

function CapacityStatus(): ReactElement {
  const [universe] = useAtom(universeAtom);
  const [selectedDate, setSelectedDate] = useState<Date | undefined | null>();
  const [selectedProvider, setSelectedProvider] =
    useState<PaxMonProviderInfo | null>(null);

  const { data: scheduleInfo } = useLookupScheduleInfoQuery();

  const request: PaxMonCapacityStatusRequest = {
    universe,
    filter_by_time: selectedDate ? "ActiveTime" : "NoFilter",
    filter_interval: {
      begin: selectedDate ? getUnixTime(selectedDate) : 0,
      end: selectedDate ? getUnixTime(add(selectedDate, { days: 1 })) : 0,
    },
  };

  const { data } = useQuery(
    queryKeys.capacityStatus(request),
    () => sendPaxMonCapacityStatusRequest(request),
    {
      enabled: selectedDate !== undefined,
    },
  );

  const scheduleRange = getScheduleRange(scheduleInfo);
  if (selectedDate === undefined && scheduleInfo) {
    setSelectedDate(scheduleRange.closestDate);
  }

  const providers: PaxMonProviderInfo[] = data
    ? data.by_provider.map((p) => p.provider_info)
    : [];

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">Kapazitätsdaten</h2>
      <div className="flex pb-2 gap-1">
        <div>
          <label>
            <span className="text-sm">Datum</span>
            <DatePicker
              value={selectedDate}
              onChange={setSelectedDate}
              min={scheduleRange.firstDay}
              max={scheduleRange.lastDay}
            />
          </label>
        </div>
        <div className="w-96">
          <label>
            <span className="text-sm">Betreiber</span>
            <ProviderComboBox
              providers={providers}
              selectedProvider={selectedProvider}
              setSelectedProvider={setSelectedProvider}
            />
          </label>
        </div>
      </div>
      <CapacityStatusDisplay data={data} selectedProvider={selectedProvider} />
      <DetailedView selectedDate={selectedDate} />
    </div>
  );
}

interface ProviderComboBoxProps {
  providers: PaxMonProviderInfo[];
  selectedProvider: PaxMonProviderInfo | null;
  setSelectedProvider: React.Dispatch<
    React.SetStateAction<PaxMonProviderInfo | null>
  >;
}

function ProviderComboBox({
  providers,
  selectedProvider,
  setSelectedProvider,
}: ProviderComboBoxProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-[450px] justify-between"
        >
          {selectedProvider
            ? selectedProvider.full_name !== ""
              ? selectedProvider.full_name
              : "Unbekannter Betreiber"
            : "Betreiber wählen..."}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[450px] p-0">
        <Command>
          <CommandInput placeholder="Betreiber suchen..." />
          <CommandList>
            <CommandEmpty>Kein Betreiber gefunden.</CommandEmpty>
            <CommandGroup>
              {providers.map((provider) => (
                <CommandItem
                  key={provider.full_name}
                  value={provider.full_name}
                  onSelect={() => {
                    setSelectedProvider(provider);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selectedProvider?.full_name === provider.full_name
                        ? "opacity-100"
                        : "opacity-0",
                    )}
                  />
                  {provider.full_name !== ""
                    ? provider.full_name
                    : "Unbekannter Betreiber"}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

interface CapacityStatusDisplayProps {
  data: PaxMonCapacityStatusResponse | undefined;
  selectedProvider: PaxMonProviderInfo | null;
}

function CapacityStatusDisplay({
  data,
  selectedProvider,
}: CapacityStatusDisplayProps) {
  if (!data) {
    return <div>Daten werden geladen...</div>;
  }

  if (!selectedProvider) {
    return <></>;
  }

  const providerData = data.by_provider.find(
    (ps) => ps.provider === selectedProvider?.full_name,
  );

  if (!providerData) {
    return <div>Betreiber nicht gefunden</div>;
  }

  return <ProviderCapacityStatus data={providerData} />;
}

interface CapacityStatusDataProps {
  data: PaxMonProviderCapacityStats;
}

function ProviderCapacityStatus({ data }: CapacityStatusDataProps) {
  interface Column {
    label: string;
    stats: PaxMonCapacityStats;
  }

  const columns: Column[] = [
    { label: "Alle Züge", stats: data.stats },
    ...data.by_category.map((c) => {
      return { label: c.category, stats: c.stats };
    }),
  ];

  return (
    <div>
      <table className="border-separate border-spacing-x-2">
        <thead>
          <tr className="text-left">
            <th className="font-medium"></th>
            {columns.map((c) => (
              <th key={c.label} className="font-medium text-center p-1">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="font-medium">Überwachte Züge</td>
            {columns.map((c) => (
              <td key={c.label} className="text-center p-1">
                {formatNumber(c.stats.tracked)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="font-medium">Kapazitätsdaten vorhanden</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.capacity_for_all_sections}
                total={c.stats.tracked}
              />
            ))}
          </tr>
          <tr>
            <td className="font-medium">Wagenreihungsdaten vorhanden</td>
            {columns.map((c) => (
              <StatsTableCell
                key={c.label}
                value={c.stats.trip_formation}
                total={c.stats.tracked}
              />
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

interface StatsTableCellProps {
  value: number;
  total: number;
}

function StatsTableCell({ value, total }: StatsTableCellProps) {
  return (
    <td
      className="text-center p-1"
      title={`${formatNumber(value)} von ${formatNumber(total)}`}
    >
      {formatPercent(value / total, {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      })}
    </td>
  );
}

interface DetailedViewProps {
  selectedDate: Date | undefined | null;
}

function DetailedView({ selectedDate }: DetailedViewProps) {
  const [active, setActive] = useState(false);

  return (
    <div className="pt-4">
      <div className="flex items-center space-x-2">
        <Switch
          id="detailed-capacity-status"
          checked={active}
          onCheckedChange={setActive}
        />
        <Label htmlFor="detailed-capacity-status">
          Detaillierte Kapazitätsstatistiken anzeigen
        </Label>
      </div>
      {active && <DetailedCapacityStatus selectedDate={selectedDate} />}
    </div>
  );
}

export default CapacityStatus;
