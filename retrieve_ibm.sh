echo "Which devices?"
read devices
for device in ${devices[@]};
do
    if [[ $device == 0 ]]; then
        remote_host="root@timecrystal.sl.cloud9.ibm.com"
    else
        remote_host="root@timecrystal${device}.sl.cloud9.ibm.com"
    fi
    remote_dir="/root/random_pauli/pauli_results/"
    local_dir="./pauli_results"
    mkdir -p "$local_dir"

    # Get remote file list
    files=$(ssh "$remote_host" "ls $remote_dir")

    for file in $files; do
        if [[ ! -f "$local_dir/$file" ]]; then
            echo "Downloading $file from $remote_host..."
            scp "$remote_host:$remote_dir/$file" "$local_dir/"
        else
            echo "Skipping $file (already exists)"
        fi
    done
done
