param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $Args
)
$env:PYTHONUTF8 = "1"
python -m streamlit $Args
