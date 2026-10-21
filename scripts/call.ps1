param([string]$Question)
$body = @{ question = $Question } | ConvertTo-Json
try {
    Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
} catch {
  if ($_.Exception.Response) {
      "HTTP $($_.Exception.Response.StatusCode.value__)"
      $_.ErrorDetails.Message
  } else {
      "No response: $($_.Exception.Message)"
  }
}