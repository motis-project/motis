import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import {
  ChevronFirst,
  ChevronLast,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useMemo } from "react";

import { TripServiceInfo } from "@/api/protocol/motis.ts";

import { formatNumber } from "@/data/numberFormat.ts";

import { formatDateTime, formatTime } from "@/util/dateFormat.ts";

import {
  EvalResult,
  TripEvalSectionInfo,
} from "@/components/eval/forecast/workerMessages.ts";
import { Button } from "@/components/ui/button.tsx";

import { cn } from "@/lib/utils.ts";

interface EvalResultsProps {
  result: EvalResult;
}

interface SectionData {
  sd: TripEvalSectionInfo;
  tsi: TripServiceInfo;
}

function ResultHeader({ result }: EvalResultsProps) {
  return (
    <div>
      <div>
        <span className="font-semibold">Betrachteter Zeitraum: </span>
        {formatDateTime(result.intervalStart)} bis{" "}
        {formatDateTime(result.intervalEnd)}
      </div>
      <div>
        <span className="font-semibold">
          Vergleich 50 %-Quantil RSL-Prognose vs. Mittelwert Reisendenzähldaten:{" "}
        </span>
        <span>MAE {formatNumber(result.q50Mae)} </span>
        <span>MSE {formatNumber(result.q50Mse)}</span>
      </div>
    </div>
  );
}

export function EvalResults({ result }: EvalResultsProps) {
  const sectionData = useMemo<SectionData[]>(
    () =>
      result.trips.flatMap((trip) =>
        trip.sections.map((sd) => {
          return { sd, tsi: trip.tsi };
        }),
      ),
    [result],
  );

  const columns = useMemo(() => {
    const columnHelper = createColumnHelper<SectionData>();
    return [
      columnHelper.accessor((row) => row.tsi.service_infos[0].category, {
        id: "category",
        header: "Kategorie",
      }),
      columnHelper.accessor("tsi.trip.train_nr", { header: "Zugnr." }),
      columnHelper.group({
        header: "Fahrtabschnitt",
        columns: [
          columnHelper.group({
            header: "Von",
            columns: [
              columnHelper.accessor("sd.from.name", {
                header: "Station",
              }),
              columnHelper.accessor("sd.departureCurrentTime", {
                header: "Abfahrt",
                cell: (info) => formatTime(info.getValue() as number),
              }),
            ],
          }),
          columnHelper.group({
            header: "Nach",
            columns: [
              columnHelper.accessor("sd.to.name", {
                header: "Station",
              }),
              columnHelper.accessor((row) => row.sd.arrivalCurrentTime, {
                header: "Abfahrt",
                cell: (info) => formatTime(info.getValue() as number),
              }),
            ],
          }),
          columnHelper.accessor("sd.duration", {
            header: "Dauer",
            cell: (info) => `${info.getValue()} Min`,
          }),
        ],
      }),
      columnHelper.group({
        header: "Reisendenzähldaten",
        columns: [
          columnHelper.accessor("sd.checkCount", { header: "Ktr." }),
          columnHelper.accessor("sd.checkPaxMin", { header: "Min" }),
          columnHelper.accessor("sd.checkPaxAvg", { header: "Avg" }),
          columnHelper.accessor("sd.checkPaxMax", { header: "Max" }),
        ],
      }),
      columnHelper.group({
        header: "RSL-Prognose",
        columns: [
          columnHelper.accessor("sd.forecastPaxQ5", { header: "5 %" }),
          columnHelper.accessor("sd.forecastPaxQ50", { header: "50 %" }),
          columnHelper.accessor("sd.forecastPaxQ95", { header: "95 %" }),
        ],
      }),
      columnHelper.group({
        header: "Vergleich",
        columns: [
          columnHelper.group({
            header: "50 % / Avg",
            columns: [
              columnHelper.accessor("sd.q50Diff", { header: "Diff" }),
              columnHelper.accessor("sd.q50Factor", {
                header: "Faktor",
                cell: (info) => formatNumber(info.getValue() as number),
              }),
            ],
          }),
        ],
      }),
    ];
  }, []);

  const table = useReactTable({
    data: sectionData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 20 } },
  });

  return (
    <>
      <ResultHeader result={result} />
      <div className="mt-4">
        <table className="text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id} colSpan={header.colSpan} className="px-2">
                    {header.isPlaceholder ? null : (
                      <>
                        <div
                          className={cn(
                            header.subHeaders.length == 0
                              ? "text-left"
                              : "text-center",
                            header.column.getCanSort() &&
                              "cursor-pointer select-none",
                          )}
                          onClick={
                            header.column.getCanSort()
                              ? header.column.getToggleSortingHandler()
                              : undefined
                          }
                        >
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                          {{
                            asc: " 🔼",
                            desc: " 🔽",
                          }[header.column.getIsSorted() as string] ?? null}
                        </div>
                      </>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-2">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => table.setPageIndex(0)}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronFirst className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            onClick={() => table.setPageIndex(table.getPageCount() - 1)}
            disabled={!table.getCanNextPage()}
          >
            <ChevronLast className="h-4 w-4" />
          </Button>
          <span className="mx-5 flex items-center gap-1">
            <div>Seite</div>
            <strong>
              {table.getState().pagination.pageIndex + 1} von{" "}
              {table.getPageCount()}
            </strong>
          </span>
          <select
            className="rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            value={table.getState().pagination.pageSize}
            onChange={(e) => {
              table.setPageSize(Number(e.target.value));
            }}
          >
            {[10, 20, 30, 40, 50].map((pageSize) => (
              <option key={pageSize} value={pageSize}>
                {pageSize} pro Seite
              </option>
            ))}
          </select>
        </div>
      </div>
    </>
  );
}
