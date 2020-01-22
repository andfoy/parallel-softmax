defmodule Barrier do

  def synchronize([_|_] = pids) do
    pids
    |> Enum.map(fn pid ->
      receive do
        {:ready, ^pid} ->
          :ok
      end
    end)

    pids
    |> Enum.map(fn pid -> send(pid, :ready) end)
  end

  def synchronize(caller) do
    send(caller, {:ready, self()})
    receive do
      :ready ->
        :ok
        # code
    end
  end
end
